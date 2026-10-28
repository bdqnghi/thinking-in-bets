"""Pilot launchers (SWE-Together Stage 1, SWE-Interact): live concurrency caps (``--caps-file``).
No trials are run."""
from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from fsa.redo import caps as caps_lib
from fsa.redo.swe_interact import pilot as swi_pilot
from fsa.redo.swe_together.pilot import LiveCap, apply_caps, live_caps, read_caps_file, watch_caps


def test_read_caps_file(tmp_path):
    p = tmp_path / "caps.json"
    assert read_caps_file(p) is None
    p.write_text(json.dumps({"S": 4, "max_total": 14}))
    assert read_caps_file(p) == {"S": 4, "max_total": 14}
    for bad in ('{"X": 1}', '{"S": -1}', '{"S": 2.5}', '{"S": true}', "[]", "{}", "{not json"):
        p.write_text(bad)
        with pytest.raises(ValueError):
            read_caps_file(p)


def test_live_cap_limits_and_raises():
    async def go():
        cap = LiveCap(1)
        order, running, peak = [], [0], [0]

        async def job(i):
            async with cap:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
                order.append(i)
                await asyncio.sleep(0.01)
                running[0] -= 1

        tasks = [asyncio.create_task(job(i)) for i in range(6)]
        await asyncio.sleep(0.015)
        assert peak[0] == 1
        await cap.set(3)                      # raising wakes waiters at once
        await asyncio.gather(*tasks)
        return order, peak[0]

    order, peak = asyncio.run(go())
    assert order == list(range(6))            # arrival order, as asyncio.Semaphore
    assert peak == 3


def test_live_cap_lowering_never_stops_holders():
    async def go():
        cap = LiveCap(2)
        release = asyncio.Event()
        started = []

        async def job(i):
            async with cap:
                started.append(i)
                await release.wait()

        tasks = [asyncio.create_task(job(i)) for i in range(3)]
        await asyncio.sleep(0.01)
        assert started == [0, 1]
        await cap.set(1)
        assert cap.active == 2                # both holders keep running
        release.set()
        await asyncio.gather(*tasks)
        return started, cap.active

    started, active = asyncio.run(go())
    assert started == [0, 1, 2] and active == 0


def test_watch_caps_applies_changes(tmp_path, capsys):
    p = tmp_path / "caps.json"

    async def go():
        sems = {r: LiveCap(n) for r, n in (("W", 8), ("M", 4), ("S", 2))}
        total = LiveCap(14)
        w = asyncio.create_task(watch_caps(p, sems, total, None, every=0.01))
        await asyncio.sleep(0.03)
        assert live_caps(sems, total) == {"caps": {"W": 8, "M": 4, "S": 2}, "max_total": 14}  # no file: unchanged
        p.write_text(json.dumps({"S": 4}))
        await asyncio.sleep(0.05)
        after = live_caps(sems, total)
        p.write_text('{"S": "five"}')         # a bad edit keeps the caps
        await asyncio.sleep(0.05)
        kept = live_caps(sems, total)
        w.cancel()
        return after, kept

    after, kept = asyncio.run(go())
    assert after == {"caps": {"W": 8, "M": 4, "S": 4}, "max_total": 14}
    assert kept == after
    out = capsys.readouterr().out
    assert "CAPS {'caps': {'W': 8, 'M': 4, 'S': 4}, 'max_total': 14}" in out
    assert out.count("ignored") == 1          # each distinct bad edit is reported once


def test_apply_caps_partial():
    async def go():
        sems = {r: LiveCap(1) for r in ("W", "M", "S")}
        total = LiveCap(3)
        await apply_caps({"M": 5, "max_total": 9}, sems, total)
        return live_caps(sems, total)

    assert asyncio.run(go()) == {"caps": {"W": 1, "M": 5, "S": 1}, "max_total": 9}


def test_shared_reader_keys(tmp_path):
    p = tmp_path / "caps.json"
    p.write_text('{"max_rf": 1, "S": 3}')
    assert caps_lib.read_caps_file(p, swi_pilot.CAPS_KEYS) == {"max_rf": 1, "S": 3}
    with pytest.raises(ValueError):          # max_rf is not a SWE-Together cap
        read_caps_file(p)


def test_swi_slots_live_caps(tmp_path, capsys):
    slots = swi_pilot.Slots({"W": 1, "M": 1, "S": 1}, 8, 2)
    started, release = [], threading.Event()

    def job(order):
        with slots.hold(order, "S", False):
            started.append(order)
            release.wait(5)

    threads = [threading.Thread(target=job, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    time.sleep(0.1)
    assert started == [0]                     # S cap 1
    p = tmp_path / "caps.json"
    p.write_text(json.dumps({"S": 3, "max_rf": 1}))
    stop = threading.Event()
    w = threading.Thread(target=swi_pilot.watch_caps, args=(p, slots, None, stop, 0.02), daemon=True)
    w.start()
    deadline = time.time() + 10               # Slots re-checks every <= 5 s; notify_all wakes it sooner
    while len(started) < 3 and time.time() < deadline:
        time.sleep(0.05)
    assert sorted(started) == [0, 1, 2]
    assert slots.snapshot() == {"caps": {"W": 1, "M": 1, "S": 3}, "max_total": 8, "max_rf": 1}
    release.set()
    stop.set()
    for t in threads:
        t.join(5)
    assert "CAPS {'caps': {'W': 1, 'M': 1, 'S': 3}, 'max_total': 8, 'max_rf': 1}" in capsys.readouterr().out


def test_swi_slots_copy_caller_caps():
    caps = {"W": 1, "M": 1, "S": 1}
    slots = swi_pilot.Slots(caps, 8, 2)
    slots.set_caps({"M": 4})
    assert caps == {"W": 1, "M": 1, "S": 1} and slots.caps["M"] == 4


def test_live_launch_guard(tmp_path):
    import subprocess
    import sys

    from fsa.redo.swe_together.pilot import live_launch, trial_tag, wait_live_launch

    c = {"rung": "M", "arm": "commit_before", "overlay": False, "replicate": 1}
    tag = trial_tag(c, "tib-test-task.a")
    assert tag == "s1_M_commit_before_nat1_tib-test-task.a"
    assert not live_launch(tag)
    longer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "--tag", tag + "b"])
    try:
        time.sleep(0.3)
        assert not live_launch(tag)           # another trial whose tag merely starts with this one
    finally:
        longer.kill()
        longer.wait()
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)", "--tag", tag, "--workers", "1"])
    try:
        time.sleep(0.3)
        assert live_launch(tag)
        waited = asyncio.run(wait_live_launch(tag, "M/commit_before/nat1 tib-test-task.a", poll=0.2))
        assert waited > 0 and p.poll() is not None
        assert asyncio.run(wait_live_launch(tag, "x", poll=0.2)) == 0
    finally:
        p.kill()
        p.wait()


def test_trigger_slots_knob(tmp_path, capsys):
    from fsa.redo.swe_together.pilot import Knob, trigger_env

    knobs = {"trigger_slots": Knob(None)}
    assert trigger_env(knobs, True) == {} and trigger_env(None, True) == {}
    p = tmp_path / "caps.json"
    p.write_text(json.dumps({"S": 4, "trigger_slots": 11}))
    assert read_caps_file(p) == {"S": 4, "trigger_slots": 11}

    async def go():
        sems = {r: LiveCap(n) for r, n in (("W", 8), ("M", 4), ("S", 2))}
        total = LiveCap(14)
        w = asyncio.create_task(watch_caps(p, sems, total, None, every=0.01, knobs=knobs))
        await asyncio.sleep(0.05)
        on = (live_caps(sems, total, knobs), trigger_env(knobs, True), trigger_env(knobs, False))
        p.write_text(json.dumps({"S": 3}))    # absent key: the knob is unchanged
        await asyncio.sleep(0.05)
        kept = trigger_env(knobs, True)
        p.write_text(json.dumps({"trigger_slots": 0}))  # 0: back to the frozen default
        await asyncio.sleep(0.05)
        off = (live_caps(sems, total, knobs), trigger_env(knobs, True))
        w.cancel()
        return on, kept, off

    on, kept, off = asyncio.run(go())
    assert on[0] == {"caps": {"W": 8, "M": 4, "S": 4}, "max_total": 14, "trigger_slots": 11}
    assert on[1] == {"SWT_JUDGE3_SLOTS": "11"} and on[2] == {}   # natural cells never get it
    assert kept == {"SWT_JUDGE3_SLOTS": "11"}
    assert off == ({"caps": {"W": 8, "M": 4, "S": 3}, "max_total": 14}, {})
    assert "'trigger_slots': 11}" in capsys.readouterr().out


def test_trigger_slots_reach_the_slot_count(monkeypatch, tmp_path):
    """The env the launcher sets is what the trigger's host-wide slot pool reads (raising works too)."""
    from fsa.redo.swe_together.agentic_judge import Slot

    monkeypatch.setenv("SWT_JUDGE3_SLOTS", "11")
    assert Slot(3, tmp_path).n == 11
    monkeypatch.setenv("SWT_JUDGE3_SLOTS", "1")
    assert Slot(3, tmp_path).n == 1
    monkeypatch.delenv("SWT_JUDGE3_SLOTS")
    assert Slot(3, tmp_path).n == 3
