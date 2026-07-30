"""Adversarial regression suite, part 1: spec-validation boundaries,
objective semantics, generator invariants, and solver-honesty checks.
Hostile inputs are constructed and the required behavior is asserted.

Run from the repo root:  python3 -B tests/adversarial_1.py
Exits nonzero if any check fails.
"""

import sys
import time

sys.path.insert(0, ".")

from worldsmith import compile_text, build_scene, solve, Scene
from worldsmith.spec import SpecError
from worldsmith.verify import check_scene
from worldsmith.engine import Engine

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
        check(name, True, str(e)[:60])


BASE = {
    "name": "t", "mode": "topdown", "size": [8, 5],
    "tiles": ["########", "#......#", "#......#", "#......#", "########"],
    "entities": [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [6.5, 3.5]}],
    "objectives": [{"kind": "reach", "target": "goal"}],
}


def scene_with(entities=None, objectives=None, **over):
    import copy
    d = copy.deepcopy(BASE)
    if entities:
        d["entities"] += entities
    if objectives:
        d["objectives"] += objectives
    d.update(over)
    return d


print("count parsing:")
p = compile_text("a dungeon with two locked doors and a lava moat around the goal")
check("'two locked doors' -> doors=2", p.doors == 2, f"doors={p.doors}")
p = compile_text("an arena with 5 monsters")
check("'5 monsters' -> enemies=5", p.enemies == 5, f"enemies={p.enemies}")
p = compile_text("collect all 4 treasures")
check("'4 treasures' -> coins=4", p.coins == 4, f"coins={p.coins}")
p = compile_text("a platformer with three spike pits and four coins")
check("'three pits, four coins'", p.spike_pits == 3 and p.coins == 4,
      f"pits={p.spike_pits} coins={p.coins}")

print("plates are crate-only:")
d = scene_with(entities=[{"type": "plate", "color": "blue", "pos": [3.5, 2.5]},
                         {"type": "crate", "pos": [5.5, 1.5]}])
s = Scene.from_json(d); s.validate()
eng = Engine(s)
from worldsmith.engine import Actions
eng.player.x, eng.player.y = 3.5, 2.5  # stand right on the plate
for _ in range(10):
    eng.step(Actions())
check("player standing on plate does NOT latch it", not eng.plates[0].latched)
expect_reject("plate with zero crates rejected",
              scene_with(entities=[{"type": "plate", "color": "blue", "pos": [3.5, 2.5]}]))
plan = compile_text("push the crate onto the pressure plate to open the gate")
crate_pressed = 0; solved = 0
for seed in range(10):
    sc = build_scene(plan, seed)
    r = solve(sc)
    if r.success:
        solved += 1
        pl = r.engine.plates[0]
        if pl.latched and any(abs(c.x - pl.x) < 0.35 and abs(c.y - pl.y) < 0.35
                              for c in r.engine.crates):
            crate_pressed += 1
check(f"all solved plate scenes pressed via crate ({crate_pressed}/{solved})",
      solved > 0 and crate_pressed == solved)

print("multi-door placement:")
from collections import Counter
p = compile_text("a large maze with two locked doors, collect all 4 treasures")
c = Counter(len(build_scene(p, s).entities_of("door")) for s in range(20))
check("maze doors=2: >=80% of seeds place 2", c.get(2, 0) >= 16, f"histogram={dict(c)}")
p3 = compile_text("a dungeon with 3 locked doors")
c3 = Counter(len(build_scene(p3, s).entities_of("door")) for s in range(20))
check("dungeon doors=3: >=60% of seeds place 3", c3.get(3, 0) >= 12, f"histogram={dict(c3)}")
pp = compile_text("a large platformer with 2 locked doors and spike pits")
cp = Counter(len(build_scene(pp, s).entities_of("door")) for s in range(20))
check("platformer doors=2: no zero-door seeds", cp.get(0, 0) == 0, f"histogram={dict(cp)}")
sc = build_scene(p, 0)
check("shortfall recorded in meta.features", "features" in sc.meta, str(sc.meta.get("features")))

print("out-of-bounds door:")
expect_reject("door at x==W rejected", scene_with(
    entities=[{"type": "key", "color": "red", "pos": [2.5, 1.5]},
              {"type": "door", "color": "red", "pos": [8.0, 1.5]}]))

print("degenerate objective values:")
expect_reject("collect count=-5 rejected",
              scene_with(entities=[{"type": "coin", "pos": [3.5, 1.5]}],
                         objectives=[{"kind": "collect", "item": "coin", "count": -5}]))
expect_reject("collect count=0 rejected",
              scene_with(entities=[{"type": "coin", "pos": [3.5, 1.5]}],
                         objectives=[{"kind": "collect", "item": "coin", "count": 0}]))
expect_reject("time_limit 0 rejected", scene_with(objectives=[{"kind": "time_limit", "seconds": 0}]))
expect_reject("duplicate objective ids rejected",
              scene_with(objectives=[{"kind": "reach", "target": "goal", "id": "obj_0"}]))
expect_reject("door on player spawn rejected", scene_with(
    entities=[{"type": "key", "color": "red", "pos": [2.5, 1.5]},
              {"type": "door", "color": "red", "pos": [1.5, 1.5]}]))
expect_reject("crate on player spawn rejected",
              scene_with(entities=[{"type": "crate", "pos": [1.5, 1.5]}]))

print("replay.html objective encoding:")
from worldsmith.render import record_frames
s = Scene.from_json(scene_with(entities=[{"type": "coin", "pos": [3.5, 1.5]}],
                               objectives=[{"kind": "collect", "item": "coin", "count": 1}]))
s.validate()
r = solve(s)
frames, _, _ = record_frames(s, r.trace)
check("frame 0 encodes pending as 'o'", all(st == "o" for st in frames[0]["obj"]),
      str(frames[0]["obj"]))
check("final frame encodes passed as 'p'", all(st == "p" for st in frames[-1]["obj"]),
      str(frames[-1]["obj"]))

print("malformed JSON -> SpecError (repairable), float sizes coerced:")
hostiles = [
    {"size": None}, {"size": ["8", "6"]}, {"entities": ["nope"]},
    {"entities": [{"type": "player", "pos": 3}]}, {"objectives": ["reach"]},
    {"tiles": [1, 2, 3]},
]
ok = True
for h in hostiles:
    d = dict(BASE, **h)
    try:
        Scene.from_json(d).validate()
        ok = False  # some of these could conceivably validate; require SpecError or pass-through validity
    except SpecError:
        pass
    except Exception as e:
        ok = False
        print(f"    escaped as {type(e).__name__}: {h}")
check("all hostile shapes raise SpecError (never TypeError etc.)", ok)
d = dict(BASE); d["size"] = [8.0, 5.0]
s = Scene.from_json(d); s.validate()
check("float size coerced to int", s.size == (8, 5), str(s.size))

print("solver budget:")
t0 = time.monotonic()
big = {"name": "big", "mode": "topdown", "size": [80, 80],
       "tiles": ["#" * 80] + ["#" + "." * 78 + "#"] * 78 + ["#" * 80],
       "entities": [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [78.5, 78.5]},
                    {"type": "crate", "pos": [40.5, 40.5]},
                    {"type": "plate", "color": "blue", "pos": [60.5, 60.5]}],
       "objectives": [{"kind": "reach", "target": "goal"},
                      {"kind": "press", "color": "blue"}]}
s = Scene.from_json(big); s.validate()
r = solve(s, deadline_s=30)
dt = time.monotonic() - t0
check(f"80x80 crate scene returns within budget ({dt:.1f}s)", dt < 45,
      f"result={r.success}/{r.reason}")

print("collect key objective without door:")
d = scene_with(entities=[{"type": "key", "color": "red", "pos": [4.5, 2.5]},
                         {"type": "door", "color": "red", "pos": [6.5, 1.5]}],
               objectives=[{"kind": "collect", "item": "key:red", "count": 1}])
# variant WITHOUT door: key + collect objective only
d2 = scene_with(entities=[{"type": "key", "color": "red", "pos": [4.5, 2.5]},
                          {"type": "crate", "pos": [5.5, 1.5]},
                          {"type": "plate", "color": "red", "pos": [3.5, 3.5]}],
                objectives=[{"kind": "collect", "item": "key:red", "count": 1}])
s2 = Scene.from_json(d2); s2.validate()
r2 = check_scene(s2)
check("collect key:red verifiable without matching door", r2.success, r2.reason)

print("reach non-goal entity:")
d = scene_with(entities=[{"type": "coin", "pos": [4.5, 2.5]}],
               objectives=[{"kind": "reach", "target": "coin_2"}])
s = Scene.from_json(d); s.validate()
r = check_scene(s)
check("reach entity-id objective passes", r.success, r.reason)

print("enemy patrol through wall rejected:")
wall_scene = {"name": "w", "mode": "topdown", "size": [9, 5],
              "tiles": ["#########", "#...#...#", "#...#...#", "#...#...#", "#########"],
              "entities": [{"type": "player", "pos": [1.5, 1.5]}, {"type": "goal", "pos": [7.5, 3.5]},
                           {"type": "enemy", "pos": [1.5, 2.5],
                            "path": [[1.5, 2.5], [7.5, 2.5]], "speed": 2}],
              "objectives": [{"kind": "reach", "target": "goal"}]}
expect_reject("patrol crossing wall rejected", wall_scene)

print("goal_touched rising edge:")
# a second unmet objective keeps the episode alive while the player sits on
# the goal, so a broken rising-edge guard would emit one event per tick
d = scene_with(entities=[{"type": "coin", "pos": [6.5, 1.5]}],
               objectives=[{"kind": "collect", "item": "coin", "count": 1}])
s = Scene.from_json(d); s.validate()
from worldsmith.engine import Engine as _Eng
eng = _Eng(s)
eng.player.x, eng.player.y = 6.5, 3.5  # on the goal, coin still uncollected
for _ in range(30):
    eng.step(Actions())
n = sum(1 for _, k, _ in eng.events if k == "goal_touched")
check("exactly one goal_touched per arrival (episode still running)", n == 1, f"count={n}")

print()
if FAILURES:
    print(f"SUITE 1 FAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("SUITE 1: all checks passed.")
