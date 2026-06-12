import importlib.util
import sys
from pathlib import Path

_PKG = Path(__file__).parent.parent / "magentic_proactive" / "energy.py"
_spec = importlib.util.spec_from_file_location("energy", _PKG)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["energy"] = _mod
_spec.loader.exec_module(_mod)

compute_energy = _mod.compute_energy
d_energy = _mod.d_energy
next_tick_from_score = _mod.next_tick_from_score
get_polling_interval = _mod.get_polling_interval


def test_energy_decay_at_zero_minutes():
    """刚发完消息，能量应为 1.0"""
    e = compute_energy(minutes_since_last=0)
    assert abs(e - 1.0) < 0.01


def test_energy_decay_at_60_minutes():
    """一小时后能量应低于 0.5"""
    e = compute_energy(minutes_since_last=60)
    assert e < 0.5


def test_energy_decay_at_24_hours():
    """24 小时后能量应低于 0.2"""
    e = compute_energy(minutes_since_last=24 * 60)
    assert e < 0.2


def test_energy_decay_at_72_hours():
    """三天后能量接近 0"""
    e = compute_energy(minutes_since_last=72 * 60)
    assert e < 0.05


def test_energy_always_in_range():
    """能量值始终在 [0, 1] 范围内"""
    for minutes in [0, 0.5, 10, 100, 1000, 10000]:
        e = compute_energy(minutes_since_last=minutes)
        assert 0.0 <= e <= 1.0


def test_d_energy_zero_when_full():
    """满电量时饥渴度为 0"""
    assert d_energy(1.0) == 0.0


def test_d_energy_one_when_empty():
    """零电量时饥渴度为 1"""
    assert d_energy(0.0) == 1.0


def test_next_tick_urgent():
    """高紧迫性 → 最短间隔"""
    interval = next_tick_from_score(0.8, tick_s3=1, tick_s2=2, tick_s1=3, tick_s0=4, jitter=0)
    assert interval == 1


def test_next_tick_idle():
    """低紧迫性 → 最长间隔"""
    interval = next_tick_from_score(0.1, tick_s3=1, tick_s2=2, tick_s1=3, tick_s0=4, jitter=0)
    assert interval == 4


def test_next_tick_boundary_070():
    """边界值 > 0.70 进入最高档"""
    interval = next_tick_from_score(0.71, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480, jitter=0)
    assert interval == 60


def test_next_tick_boundary_040():
    """边界值 > 0.40 进入第二档"""
    interval = next_tick_from_score(0.41, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480, jitter=0)
    assert interval == 120


def test_next_tick_boundary_020():
    """边界值 > 0.20 进入第三档"""
    interval = next_tick_from_score(0.21, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480, jitter=0)
    assert interval == 240


def test_next_tick_boundary_default():
    """边界值 ≤ 0.20 进入最低档"""
    interval = next_tick_from_score(0.20, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480, jitter=0)
    assert interval == 480


def test_next_tick_jitter_range():
    """抖动应在 ±30% 范围内"""
    base = 480
    intervals = [
        next_tick_from_score(0.1, tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=base, jitter=0.3)
        for _ in range(200)
    ]
    assert all(base * 0.7 <= i <= base * 1.3 for i in intervals)


def test_next_tick_min_one():
    """间隔最小为 1 秒"""
    interval = next_tick_from_score(0.9, tick_s3=1, tick_s2=2, tick_s1=3, tick_s0=4, jitter=0.5)
    assert interval >= 1


def test_get_polling_interval_recent():
    """刚发完消息应取最长间隔"""
    interval = get_polling_interval(
        minutes_since_last=0,
        tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480,
        jitter=0,
    )
    assert interval == 480


def test_get_polling_interval_long_absence():
    """长时间未联系——饥渴度被 weight 压低，72h 后约 0.34，不到 0.40 阈值"""
    interval = get_polling_interval(
        minutes_since_last=72 * 60,
        tick_s3=60, tick_s2=120, tick_s1=240, tick_s0=480,
        jitter=0,
    )
    # compute_energy(4320) ≈ 0.034, d_energy ≈ 0.966, base_score = 0.966*0.35 ≈ 0.338
    # 0.20 < 0.338 ≤ 0.40 → tick_s1 = 240
    assert interval == 240
