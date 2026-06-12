# DeerFlow 记忆系统 + 主动推送改造设计

## 目标

在 DeerFlow 上实现双层记忆系统（Markdown 文件 + Chroma 向量检索）和主动推送系统（电量模型 + LLM Agent 分类 + Drift 空闲任务），用于面试展示。
核心原则：遵循 DeerFlow 现有架构规范，通过新增中间件和实现 MemoryStorage 子类完成改造，不修改核心链路。

---

## 架构全景

```
                        DeerFlow 现有架构（不动）
┌──────────────────────────────────────────────────────────┐
│  IM → Gateway → lead_agent (LangGraph)                    │
│               ├─ DynamicContextMiddleware                 │
│               ├─ SkillActivationMiddleware                │
│               ├─ ... 20 个现有中间件 ...                  │
│               └─ MemoryMiddleware                        │
└──────────────────────────────────────────────────────────┘
        │                              │
        │  新增中间件（插入链中）         │  新增循环（独立运行）
        ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│ VectorRetrieval      │    │  ProactiveLoop               │
│ Middleware           │    │  ├─ 电量模型                  │
│ (DynamicContext      │    │  ├─ 三路 MCP 数据拉取          │
│  之后插入)           │    │  ├─ LLM Agent 分类            │
│                      │    │  ├─ 分级 ACK 去重             │
│ 引用:                │    │  └─ Drift 空闲引擎            │
│ • magentic-memory    │    │                              │
│   (独立算法包)        │    │  引用:                        │
│   ├─ QueryRewriter   │    │  • magentic-proactive         │
│   ├─ HyDE 生成        │    │    (独立算法包)               │
│   ├─ RRF 融合         │    │    ├─ energy.py              │
│   ├─ Chroma 检索      │    │    ├─ judge.py               │
│   └─ Consolidator    │    │    └─ drift.py                │
└──────────────────────┘    └──────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  MarkdownMemoryStore                 │
│  (MemoryStorage 子类)                │
│  {base_dir}/users/{id}/memory/       │
│  ├─ MEMORY.md    ├─ PENDING.md       │
│  ├─ HISTORY.md   ├─ SELF.md          │
│  ├─ RECENT_CONTEXT.md                │
│  └─ chroma/  (Chroma 持久化)         │
└──────────────────────────────────────┘
```

---

## 模块一：Markdown Memory Storage

### 目标
实现 DeerFlow 的 `MemoryStorage` 抽象接口，用五个 Markdown 文件替代现有的 `memory.json`。

### 文件结构
```
{base_dir}/users/{user_id}/memory/
├── MEMORY.md           # 用户画像全文，bullet 列表带 7 种 tag
├── HISTORY.md          # 时间线事件，每条 [YYYY-MM-DD HH:MM] 第三人称摘要
├── SELF.md             # Agent 自我认知，三段（人格/对用户理解/关系定位）
├── RECENT_CONTEXT.md   # 五维压缩摘要 + 近期对话预览
├── PENDING.md          # 增量写入缓冲，格式同 MEMORY.md
└── chroma/             # Chroma 向量持久化目录
```

### 文件格式

**MEMORY.md** — 全文注入 `<system-reminder>`：
```markdown
- [identity] 用户是互联网产品经理，3 年经验
- [preference] 用户不喜欢悬疑压抑风格的游戏
- [key_info] GitHub 用户名: example
- [health_long_term] 用户有慢性偏头痛
- [requested_memory] 项目 deadline 是 6 月 15 日
- [correction] 更正：用户不是学生，已毕业
- [agent_context] Web 服务运行在 8080 端口
```

**HISTORY.md** — embed 到 Chroma，不全文注入：
```markdown
[2026-06-08 14:30] 用户完成了毕业答辩
[2026-06-09 10:15] 用户面试字节跳动二面通过，面试官对记忆系统设计感兴趣
```

**RECENT_CONTEXT.md** — 注入 `<system-reminder>` 上下文帧：
```markdown
# Recent Context
## Compression
until: msg_12345
- 最近持续关注：RAG 架构设计；Chroma 选型
- 最近明确偏好：用户偏好 Vite 而非 Webpack
- 最近待延续话题：记忆系统的 HyDE 实现细节
- 最近避免事项：用户说今晚不想聊技术
## Ongoing Threads
- 用户最近持续受失眠困扰
## Recent Turns
[user] 今天聊了主动推送的电量模型
[a-preview] 三重指数衰减的核心是三个时间尺度叠加...
```

**PENDING.md** — 格式同 MEMORY.md，Consolidation 写入，Optimizer 定时归档。

### 接口定义
```python
class MarkdownMemoryStore(MemoryStorage):
    # 标准接口（兼容现有调用方）
    async def read_memory(user_id) -> dict
    async def write_memory(user_id, data) -> None

    # 新增接口
    def read_long_term(user_id) -> str            # MEMORY.md 全文
    def read_history(user_id) -> str              # HISTORY.md 全文
    def read_recent_context(user_id) -> str       # 压缩摘要（不含 Recent Turns）
    def append_pending(user_id, items: list[str]) -> None
    def read_pending(user_id) -> str
    def archive_pending(user_id, new_memory: str) -> None
    def write_self(user_id, content: str) -> None
```

### 配置
```yaml
memory:
  storage_class: magentic_memory.markdown_store.MarkdownMemoryStore
  markdown:
    base_dir: "{deer-flow}/.deer-flow"
    max_pending_items: 200
    optimizer_interval_hours: 18
```

---

## 模块二：Chroma 向量检索

### 目标
在 Chroma 中为四类记忆建立独立 collection，通过 VectorRetrievalMiddleware 在每轮对话时执行完整检索链路。

### 四类 Collection
| Collection | 来源 | 写入时机 |
|---|---|---|
| `memory_events` | HISTORY.md 的每条事件 | Consolidation 后 embed |
| `memory_procedures` | 对话中隐式提取 | Consolidation 后隐式提取 |
| `memory_preferences` | 同上 | 同上 |
| `memory_profiles` | 同上 | 同上 |

### 检索链路
```
用户消息
  │
  ├─ 1. QueryRewriter     → 轻量 LLM 消解代词 + 意图分类
  │                         超时 800ms → fail-open 回退原始查询
  │
  ├─ 2. HyDE 生成          → 轻量 LLM 生成 2 条假设文档
  │                         超时 2s → 放弃假设，单路检索
  │
  ├─ 3. 多向量通道          → 原始查询 + 假设 × 4 collection = 并行搜索
  │                         Chroma filter: {"memory_type": "event"}
  │
  ├─ 4. 关键词通道          → CJK 二元分词 + ASCII 分词 → Chroma 全文检索
  │
  ├─ 5. RRF 融合           → rrf_score = 1/(60+vec_rank) + 0.5/(60+kw_rank)
  │                         向量权重 1.0，关键词权重 0.5
  │
  └─ 6. 三段式注入          → 强制约束 > 规范 > 历史事件
                             1200 字符预算
                             置信度标注（"有印象，不确定"）
```

### 注入格式
```
<system-reminder>
以下记忆条目来自系统检索，不是用户陈述：

【强制约束】
- [proc_001] 查菜谱只给 20 分钟以内的（必须调用工具：web_search）

【操作规范】
- [pref_002] 用户偏好日式 RPG，讨厌悬疑类游戏

【相关历史】
- [event_003] 用户 6 月 9 日面试字节二面 | 距今 3 天 | 证据: 可回源原文
- [event_005] 用户上周通关了 Persona 5 | 有印象，不确定
</system-reminder>
```

### 中间件插入位置
在 `build_middlewares()` 中，`DynamicContextMiddleware`（位置 3）和 `SkillActivationMiddleware`（位置 4）之间插入 `VectorRetrievalMiddleware`。

### 配置
```yaml
memory:
  vector:
    enabled: true
    embedding_model: text-embedding-3-small
    persist_dir: "{deer-flow}/.deer-flow/users/{user_id}/memory/chroma"
    hyde_enabled: true
    hyde_timeout_ms: 2000
    rrf_k: 60
    rrf_keyword_weight: 0.5
    score_threshold: 0.45
    inject_max_chars: 1200
```

---

## 模块三：PENDING 缓冲 + Optimizer

### 目标
将高频增量写入（Consolidation 每几轮触发）与低频全量更新（MEMORY.md 全文注入 prompt）解耦，保护 prompt 缓存稳定性。

### 写入链路改造
```
现有: MemoryMiddleware → MemoryUpdateQueue(30s) → MemoryUpdater → memory.json

改造: MemoryMiddleware → MemoryUpdateQueue(30s) → ConsolidationUpdater
                                                      │
                                                 ┌────┴────┐
                                                 ▼         ▼
                                            PENDING.md  Chroma
                                              (缓冲)    (实时)
                                                 │
                                                 │ 每 18h
                                                 ▼
                                          MarkdownOptimizer
                                                 │
                                          LLM 合并去重
                                                 │
                                          ┌──────┴──────┐
                                          ▼              ▼
                                      MEMORY.md      SELF.md
```

### ConsolidationUpdater
替代现有 `MemoryUpdater`，改 LLM prompt 输出格式：
- `history_entries[]` → 追加 HISTORY.md + embed 到 Chroma
- `pending_items[]` → 追加 PENDING.md（7 种 tag）
- `recent_context` → 覆盖 RECENT_CONTEXT.md

### MarkdownOptimizer
每隔 `optimizer_interval_hours`（默认 18h）运行的定时任务：

1. 原子快照：`PENDING.md` → `PENDING.snapshot.md`（POSIX rename）
2. Memory 合并：LLM 读 PENDING + MEMORY → 新 MEMORY.md（事实去重、分类、置信度排序）
3. Self 更新：LLM 读 PENDING + SELF → 新 SELF.md（三段自我认知）
4. 提交：写 MEMORY.md + SELF.md，删除 snapshot
5. 失败回滚：snapshot 内容合并回 PENDING.md

---

## 模块四：主动推送系统

### 目标
实现独立于被动回复循环的 ProactiveLoop，包含电量模型、数据拉取、LLM 分类决策和分级 ACK 去重。

### 电量模型
```
E(t) = 0.50·e^(-t/30min) + 0.35·e^(-t/240min) + 0.15·e^(-t/2880min)
D_energy = 1 - E
base_score = D_energy × 0.35

base_score > 0.70 → 1min 间隔
base_score > 0.40 → 2min 间隔
base_score > 0.20 → 4min 间隔
else              → 8min 间隔

final_interval = base × uniform(0.7, 1.3)  # ±30% 随机抖动
```

### 数据拉取
复用 DeerFlow 现有 MCP server 连接，从 `proactive_sources.json` 读配置：

| 通道 | 内容 | 处理 |
|---|---|---|
| alert | 高优先级告警 | 直接透传，不评分 |
| content | RSS、新闻、GitHub | LLM 逐条分类 |
| context | 背景（天气、股价） | 概率注入作兜底 |

### LLM Agent 分类
LLM 以 Agent 身份运行，调用工具分类内容：

1. 接收系统提示词（分类规则）+ 上下文帧（数据 + 记忆 + 偏好）
2. 逐条审视内容 → 调用 `mark_interesting(item_id, reason)` 或 `mark_not_interesting(item_id)`
3. 完成分类 → 决策：
   - 无兴趣条目 → `finish_turn(decision="skip")`
   - 有兴趣条目 → `message_push(draft)` 拟稿 → `finish_turn(decision="reply")`
4. 完备性守卫：未分类条目自动提示补充
5. 反思守卫：有兴趣条目但未 finish 时强制提示收尾

### 分级 ACK 去重
| 场景 | TTL |
|---|---|
| 推送消息中引用了该内容 | 168h（7天） |
| 内容评分低没推送 | 24h |
| 判定为不相关 | 720h（30天） |

Delivery 判重：`SHA1(排序后的引用记忆 ID 列表)`

### 与 DeerFlow 的对接
- 推送通道复用 `ChannelManager.send_message()`
- 数据源复用 DeerFlow MCP server 配置
- 作为 `asyncio.create_task` 在 Gateway 启动时创建

---

## 模块五：Drift 空闲任务引擎

### 目标
当主动推送所有数据源均为空时，不空转，执行 SKILL.md 分步后台任务。

### 触发条件
- 三路数据源（alert / content / context）均为空
- 距上次 Drift 超过 `drift_min_interval_hours`（默认 3h）

### 执行流程
```
ProactiveLoop 无内容 → cooldown 检查 → DriftTurnPipeline
  ├─ Scan:   扫描 drift/skills/*/SKILL.md，过滤 MCP 依赖不满足的
  ├─ Prepare: 构建受限工具集（只读文件、记忆召回、shell、message_push）
  ├─ Execute: LLM 按 SKILL.md 分步执行，每步调一个工具，最多 20 步
  │           调 message_push 后限制为只写文件
  └─ Finish:  记录 skill_used、摘要、下一步建议
```

### SKILL.md 格式
```markdown
---
name: 记忆健康检查
description: 审计长期记忆的准确性和一致性
requires_mcp: []
---

## 目标
检查 MEMORY.md 和 HISTORY.md 中是否存在矛盾或过时条目。

## 步骤
1. 读 MEMORY.md 全文
2. 读 HISTORY.md 最近 30 天条目
3. 逐条检查一致性
4. 发现矛盾 → 写 correction tag
5. 发现过时 → 标记失效
6. 完成 → 输出审计报告
```

### 与 DeerFlow skills 的关系
DeerFlow 已有 skills 机制（Markdown 定义工作流），但仅用于用户主动触发。Drift 复用 SKILL.md 格式，但执行时机为"空闲时自动执行"。

---

## 文件结构总览

```
deer-flow/
├── backend/
│   ├── packages/
│   │   └── harness/
│   │       └── deerflow/
│   │           └── agents/
│   │               ├── middlewares/
│   │               │   ├── vector_retrieval_middleware.py    # 新增
│   │               │   └── proactive_loop_middleware.py      # 新增
│   │               └── memory/
│   │                   ├── markdown_store.py                 # 新增: MemoryStorage 子类
│   │                   ├── consolidation_updater.py          # 新增: 替代 MemoryUpdater
│   │                   └── markdown_optimizer.py             # 新增: 18h 定时归档
│   └── magentic-packages/
│       ├── magentic-memory/                                   # 独立算法包
│       │   ├── vector_store.py       (Chroma 封装)
│       │   ├── retriever.py          (HyDE + RRF + 查询重写)
│       │   └── consolidator.py       (PENDING 缓冲 + Optimizer)
│       └── magentic-proactive/                                # 独立算法包
│           ├── energy.py             (电量模型衰减)
│           ├── judge.py              (LLM Agent 分类)
│           └── loop.py               (自适应轮询)
├── config.yaml                        # 扩展: memory.markdown, memory.vector, proactive
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-12-memory-and-proactive-design.md
```

---

## 错误处理

### 检索链路
- QueryRewriter 超时或解析失败 → 回退原始查询，不阻断
- HyDE 生成超时 → 放弃假设文档，单路检索
- Chroma 连接失败 → 跳过向量检索，只返回 Markdown 全文注入结果
- RRF 输入为空 → 直接返回非空通道的结果

### 写入链路
- PENDING 快照失败 → 放弃本轮合并，保留 PENDING 等下一轮
- Optimizer LLM 返回格式异常 → 回滚 snapshot 内容到 PENDING
- Chroma embed 失败 → 记录错误日志，不阻塞 PENDING 写入

### 主动推送
- MCP 连接断开 → McpClientPool 自动重连
- LLM 分类超时 → 本轮 skip，等下一 tick
- ACK 调用失败 → 记录日志，本地去重表仍生效

---

## 测试策略

### 单元测试
- `MarkdownMemoryStore` 五个文件的读写、追加、归档
- `ChromaVectorStore` 四类 collection 的 CRUD 和元数据过滤
- `QueryRewriter` 各种输入 → 期望改写
- `energy.py` 不同时间间隔 → 期望电量值
- `ConsolidationUpdater` LLM 输出解析

### 集成测试
- `VectorRetrievalMiddleware` 在中间件链中的插入位置和调用顺序
- `ProactiveLoop` 与 DeerFlow ChannelManager 的并发安全
- Optimizer 定时触发和快照回滚

### 端到端测试
- 完整一轮对话：消息到达 → 向量检索 → 注入 → LLM 回复 → Consolidation → PENDING 写入
- 完整一轮 tick：电量计算 → MCP 拉取 → LLM 分类 → 推送/Drift → ACK

---

## 面试叙事建议

**项目名**：DeerFlow 记忆与主动推送系统改造

**一句话**：在字节跳动 70k star 的开源 Agent 框架上，深入改造了记忆系统和主动推送架构。

**展开点**：

1. **架构决策**："DeerFlow 有 23 个中间件组成的链式扩展体系，我利用了他们的 MemoryStorage 抽象类和中间件链，在不修改核心 LangGraph 图的前提下完成了改造。"

2. **双层记忆**："记忆层用了 Markdown 全文注入 + Chroma 向量检索互补。Markdown 让 LLM 理解完整用户画像，Chroma 用 HyDE 假设嵌入 + RRF 融合做精准召回。中间加了一层 PENDING 缓冲保 prompt 缓存——高频增量写 PENDING，18 小时一次批量归档到 MEMORY.md。"

3. **主动推送**："从 magentic-agent 参考了三重指数衰减的电量模型，让轮询频率根据用户活跃度自适应变化。不是机械定时——你先聊完它不烦你，半天没动静才加速。LLM 自己用 mark_interesting/mark_not_interesting 工具逐条分类，不打分。"

4. **空闲机制**："没内容推的时候不空转，执行 SKILL.md 定义的后台任务——记忆审计、画像补全、自我诊断。"

5. **技术深度**："写入了 HyDE 的完整实现（轻量 LLM 生成假设文档 → 多向量通道并行检索 → 非破坏性 union 去重）、RRF 排序融合（向量权重 1.0 / 关键词权重 0.5 / k=60）、查询重写管道（800ms 超时 + fail-open）、以及分级 ACK 去重体系（7 天/1 天/30 天三级 TTL）。"
