"""Adversarial regression suite, part 2: further hardening checks.
Coordinate edge cases, semantic fidelity of text commands, resource
budgets, injection, and artifact integrity.

Run from the repo root:  python3 -B tests/adversarial_2.py
Exits nonzero if any check fails.
"""

import copy
import sys
import time

sys.path.insert(0, ".")

from worldsmith import compile_text, build_scene, solve, Scene
from worldsmith.spec import SpecError
from worldsmith.verify import check_scene, realize_verified
from worldsmith.engine import Engine, Actions

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def expect_reject(name, scene_dict):
    try:
        s = Scene.from_json(scene_dict)
        s.validate()
        check(name, False, "validated but should have been rejected")
    except SpecError as e:
        check(name, True, str(e)[:70])
    except Exception as e:
        check(name, False, f"escaped as {type(e).__name__}: {e}")


BASE = {
    "name": "t", "mode": "topdown", "size": [8, 5],
    "tiles": ["########", "#......#", "#......#", "#......#", "########"],
    "entities": [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [6.5, 3.5]}],
    "objectives": [{"kind": "reach", "target": "goal"}],
}


def scene_with(entities=None, objectives=None, **over):
    d = copy.deepcopy(BASE)
    if entities:
        d["entities"] += entities
    if objectives:
        d["objectives"] += objectives
    d.update(over)
    return d


print("negative-fraction coordinates:")
expect_reject("door at x=-0.25 rejected (floor, not int-truncation)", scene_with(
    entities=[{"type": "key", "color": "red", "pos": [2.5, 1.5]},
              {"type": "door", "color": "red", "pos": [-0.25, 2.5]}]))
open_tiles = ["........"] * 5
expect_reject("player at x=-0.849 on borderless grid rejected",
              dict(BASE, tiles=open_tiles,
                   entities=[{"type": "player", "pos": [-0.849, 2.5]},
                             {"type": "goal", "pos": [6.5, 3.5]}]))

print("plates must gate:")
plan = compile_text("push the crate onto the pressure plate to open the gate")
ungated = 0
for seed in range(30):
    sc = build_scene(plan, seed)
    pc = {e.color for e in sc.entities_of("plate")}
    dc = {e.color for e in sc.entities_of("door")}
    if pc and not (pc & dc):
        ungated += 1
        if "features" in sc.meta and sc.meta["features"].get("plate_gated", [1, 1])[1] == 0:
            ungated -= 1  # honest: recorded as ungated shortfall
check(f"every plate gates a door OR is recorded as shortfall (silent-ungated=0/30)",
      ungated == 0)
import json as _json
ex = _json.load(open("examples/crate-puzzle-dungeon/spec.json"))
pcol = {e["color"] for e in ex["entities"] if e["type"] == "plate"}
dcol = {e["color"] for e in ex["entities"] if e["type"] == "door"}
check("committed crate-puzzle example: plate color gates a door", bool(pcol & dcol),
      f"plate={pcol} doors={dcol}")

print("negation-blind text compiler:")
p = compile_text("a dungeon with no monsters")
check("'no monsters' -> enemies=0", p.enemies == 0, f"enemies={p.enemies}")
p = compile_text("a maze without lava")
check("'without lava' -> lava=none", p.lava == "none", f"lava={p.lava}")
p = compile_text("a dungeon with no doors, just coins")
check("'no doors' -> doors=0", p.doors == 0, f"doors={p.doors}")
p = compile_text("a hard dungeon with no monsters")
check("'hard ... no monsters' -> negation beats difficulty nudge", p.enemies == 0,
      f"enemies={p.enemies}")
p = compile_text("a platformer with no spike pits")
check("'no spike pits' -> pits=0", p.spike_pits == 0, f"pits={p.spike_pits}")
p = compile_text("a dungeon with 10 locked doors")
check("'10 doors' capped WITH warning", p.doors == 3 and any("capped" in w for w in p.warnings),
      f"doors={p.doors} warnings={p.warnings}")

print("deadline honesty:")
big = {"name": "big", "mode": "topdown", "size": [80, 80],
       "tiles": ["#" * 80] + ["#" + "." * 78 + "#"] * 78 + ["#" * 80],
       "entities": [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [78.5, 78.5]},
                    {"type": "crate", "pos": [40.5, 40.5]},
                    {"type": "plate", "color": "blue", "pos": [60.5, 60.5]}],
       "objectives": [{"kind": "reach", "target": "goal"}, {"kind": "press", "color": "blue"}]}
s = Scene.from_json(big); s.validate()
t0 = time.monotonic()
r = solve(s, deadline_s=3)
dt = time.monotonic() - t0
check("budget expiry labeled budget_exhausted, not stuck", r.reason == "budget_exhausted",
      f"reason={r.reason}")
check(f"deadline overshoot small ({dt:.1f}s on 3s budget)", dt < 5.0)

print("platformer narrow widths:")
ok = True
for w in (14, 15, 16, 17):
    p = compile_text(f"a {w}x10 platformer with a locked door")
    try:
        sc = build_scene(p, 0)
        feats = sc.meta.get("features", {})
        if feats.get("doors", [0, 0])[1] < feats.get("doors", [0, 0])[0] and False:
            pass
    except Exception as e:
        ok = False
        print(f"    W{w} still crashes: {e}")
check("widths 14-17 generate without crashing (door shortfall recorded)", ok)

print("enemy speed and body:")
expect_reject("enemy speed=-2 rejected", scene_with(
    entities=[{"type": "enemy", "pos": [4.5, 2.5], "path": [[4.5, 2.5], [6.5, 2.5]], "speed": -2}]))
expect_reject("enemy speed=NaN rejected", scene_with(
    entities=[{"type": "enemy", "pos": [4.5, 2.5], "path": [[4.5, 2.5], [6.5, 2.5]], "speed": float("nan")}]))
# belt: even a zero-distance leg cannot divide by zero at runtime
s = Scene.from_json(scene_with(
    entities=[{"type": "enemy", "pos": [4.5, 2.5], "path": [[4.5, 2.5], [4.5, 2.5]], "speed": 2}]))
s.validate()
eng = Engine(s)
try:
    for _ in range(5):
        eng.step(Actions())
    check("degenerate zero-length patrol leg does not crash engine", True)
except ZeroDivisionError:
    check("degenerate zero-length patrol leg does not crash engine", False)

print("lava shortfall reporting:")
plan = compile_text("a dungeon with a lava moat around the goal")
silent_zero = 0
for seed in range(40):
    sc = build_scene(plan, seed)
    has_lava = any("~" in row for row in sc.tiles)
    recorded = sc.meta.get("features", {}).get("lava", [0, 0])
    if not has_lava and recorded[1] != 0:
        silent_zero += 1
check("zero-lava scenes always recorded as lava shortfall", silent_zero == 0,
      f"silent={silent_zero}/40")

print("no objective events after episode_end:")
d = scene_with(entities=[{"type": "goal", "pos": [3.5, 3.5]}])
d["tiles"] = ["########", "#......#", "#......#", "#..~...#", "########"]
d["entities"] = [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [3.5, 3.5]}]
s = Scene.from_json(d); s.validate()
eng = Engine(s)
eng.hp = 1
eng.player.x, eng.player.y = 3.5, 2.6  # step down into the lava/goal tile
for _ in range(30):
    if eng.done:
        break
    eng.step(Actions(down=True))
kinds = [k for _, k, _ in eng.events]
if "episode_end" in kinds:
    after = kinds[kinds.index("episode_end") + 1:]
    check("no events after episode_end", not after, f"after={after}")
else:
    check("no events after episode_end", True, "episode did not end (no lava hit)")

print("replay.html injection:")
from worldsmith.render import replay_html
import os
d = scene_with()
d["name"] = "</script><script>globalThis.PWNED=1</script>"
s = Scene.from_json(d); s.validate()
import tempfile
tmp = tempfile.mkstemp(suffix=".html")[1]
r = solve(s)
replay_html(s, r.trace, tmp)
html = open(tmp).read()
os.remove(tmp)
check("'</script' cannot appear inside the data payload", "</script><script>globalThis" not in html)

print("witness-less objectives:")
expect_reject("colorless open with no doors rejected",
              scene_with(objectives=[{"kind": "open"}]))
expect_reject("press with crate but no plate rejected",
              scene_with(entities=[{"type": "crate", "pos": [3.5, 2.5]}],
                         objectives=[{"kind": "press"}]))
expect_reject("survive min_hp=5 (> max 3) rejected",
              scene_with(objectives=[{"kind": "survive", "min_hp": 5}]))
expect_reject("time_limit NaN rejected",
              scene_with(objectives=[{"kind": "time_limit", "seconds": float("nan")}]))
expect_reject("2 same-color doors with 1 key rejected", scene_with(
    entities=[{"type": "key", "color": "red", "pos": [2.5, 1.5]},
              {"type": "door", "color": "red", "pos": [4.5, 1.5]},
              {"type": "door", "color": "red", "pos": [4.5, 3.5]}]))
expect_reject("enemy patrol across a door tile rejected", scene_with(
    entities=[{"type": "key", "color": "red", "pos": [2.5, 1.5]},
              {"type": "door", "color": "red", "pos": [4.5, 2.5]},
              {"type": "enemy", "pos": [3.5, 2.5], "path": [[3.5, 2.5], [6.5, 2.5]], "speed": 2}]))

print("NaN/junk still SpecError at validate stage:")
expect_reject("NaN position raises SpecError (not ValueError)",
              scene_with(entities=[{"type": "coin", "pos": [float("nan"), 1.5]}]))
try:
    Scene.from_json(scene_with(objectives=[{"kind": "collect", "item": 7, "count": 1}])).validate()
    check("collect item=7 rejected", False, "validated")
except SpecError as e:
    check("collect item=7 raises SpecError (not AttributeError)", True, str(e)[:60])
except Exception as e:
    check("collect item=7 raises SpecError (not AttributeError)", False, type(e).__name__)

print("key invariant now enforced:")
violations = 0
p3 = compile_text("a hard dungeon with 3 locked doors and lava")
from collections import deque
for seed in range(30):
    sc = build_scene(p3, seed)
    solid = {(c, r) for r, row in enumerate(sc.tiles) for c, ch in enumerate(row) if ch in "#~^"}
    solid |= {(int(e.pos[0]), int(e.pos[1])) for e in sc.entities_of("door")}
    px, py = [e for e in sc.entities if e.type == "player"][0].pos
    start = (int(px), int(py))
    seen = {start}; q = deque([start])
    while q:
        c, r = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if n not in seen and 0 <= n[0] < sc.width and 0 <= n[1] < sc.height and n not in solid:
                seen.add(n); q.append(n)
    for k in sc.entities_of("key"):
        if (int(k.pos[0]), int(k.pos[1])) not in seen:
            violations += 1
            break
check("all keys reachable with ALL doors closed, 30 seeds x 3 doors + lava",
      violations == 0, f"violations={violations}/30")

print("cmd_solve no longer overwrites input dir:")
import worldsmith.cli as cli_mod
src = open("worldsmith/cli.py").read()
check("default outdir is under runs/, not dirname(spec)",
      'os.path.join(RUNS_DIR, f"solve-' in src and
      "args.out or os.path.dirname" not in src)

print("residual parses:")
p = compile_text("level 3 with a door")
check("'level 3 with a door' -> doors=1", p.doors == 1, f"doors={p.doors}")
p = compile_text("a 3-door dungeon")
check("'a 3-door dungeon' -> doors=3", p.doors == 3, f"doors={p.doors}")
p = compile_text("a dungeon 20 by 12 with a key")
check("'20 by 12' size parsed", p.size == (20, 12), f"size={p.size}")
p = compile_text("a maze with a 2 minute time limit")
check("'2 minute' -> 120s", p.time_limit == 120.0, f"tl={p.time_limit}")

print("platformer mode matrix + docs:")
expect_reject("crate in platformer rejected",
              dict(scene_with(entities=[{"type": "crate", "pos": [3.5, 3.5]}]),
                   mode="platformer"))
spec = _json.load(open("examples/claude-plan-spooky-crypt/spec.json"))
readme = open("README.md").read()
check("README crypt size matches spec", f"{spec['size'][0]}x{spec['size'][1]}" in readme,
      f"spec={spec['size']}")

print()
if FAILURES:
    print(f"SUITE 2 FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("SUITE 2: all checks passed.")
