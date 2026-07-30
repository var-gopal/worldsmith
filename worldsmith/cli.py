"""WorldSmith command-line interface.

    python3 -m worldsmith run "a dungeon with two locked doors and a lava moat"
    python3 -m worldsmith run "a platformer with spike pits and 4 coins" --seed 7
    python3 -m worldsmith gen "..." --brain llm
    python3 -m worldsmith solve runs/<name>/spec.json
    python3 -m worldsmith play runs/<name>/spec.json
    python3 -m worldsmith batch 20 "a maze with a key"
    python3 -m worldsmith demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .engine import Engine, Actions
from .procgen import GenPlan, build_scene
from .render import ascii_frame, scene_png, replay_html, replay_gif
from .solver import solve
from .spec import Scene, SpecError
from .textgen import compile_text
from .verify import realize_verified, check_scene

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

DEMO_PROMPTS = [
    ("a small dungeon with a locked door and three coins", 1),
    ("a large maze with two locked doors, collect all 4 treasures", 2),
    ("an arena with two patrolling guards and a lava moat around the goal", 3),
    ("push the crate onto the pressure plate to open the gate", 2),
    ("a hard dungeon: two keys, lava, guards, 60 second time limit", 5),
    ("a platformer with three spike pits and four coins", 6),
    ("a long platformer with a locked door and treacherous gaps", 7),
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="worldsmith",
        description="Text -> verified-playable 2D game environments, plus an agent that beats them.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_gen_args(p):
        p.add_argument("text", help="what to build, in plain English")
        p.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
        p.add_argument("--brain", default="rules",
                       choices=("rules", "llm", "llm-scene", "claude", "claude-scene"),
                       help="text->scene backend: offline rules (default), llm plan, llm full-scene "
                            "(claude/claude-scene are aliases; set WORLDSMITH_LLM_CMD to use any LLM CLI)")
        p.add_argument("--out", default=None, help="output directory (default runs/<name>-s<seed>)")

    p_run = sub.add_parser("run", help="generate + verify + agent playthrough + artifacts")
    add_gen_args(p_run)
    p_run.add_argument("--player", choices=("planner", "llm", "claude"), default="planner",
                       help="who plays the showcased episode (llm = an LLM plays from ASCII; "
                            "claude is an alias)")

    p_gen = sub.add_parser("gen", help="generate + verify only (no showcased playthrough)")
    add_gen_args(p_gen)

    p_solve = sub.add_parser("solve", help="run the solver agent on an existing spec.json")
    p_solve.add_argument("spec")
    p_solve.add_argument("--player", choices=("planner", "llm", "claude"), default="planner")
    p_solve.add_argument("--out", default=None)

    p_play = sub.add_parser("play", help="play a spec yourself in the terminal (WASD + space)")
    p_play.add_argument("spec")

    p_verify = sub.add_parser(
        "verify", help="replay a run's recorded trace and check it reproduces result.json")
    p_verify.add_argument("rundir", help="a run directory containing spec.json, trace.json, result.json")

    p_show = sub.add_parser("show", help="print a spec as ASCII")
    p_show.add_argument("spec")

    p_batch = sub.add_parser("batch", help="mass-produce N verified environments from one command")
    p_batch.add_argument("n", type=int)
    add_gen_args(p_batch)

    sub.add_parser("demo", help="run a curated set of prompts end to end")

    args = ap.parse_args(argv)
    handlers = {"run": cmd_run, "gen": cmd_gen, "solve": cmd_solve, "play": cmd_play,
                "show": cmd_show, "batch": cmd_batch, "demo": cmd_demo,
                "verify": cmd_verify}
    try:
        return handlers[args.cmd](args)
    except SpecError as e:
        print(f"invalid spec: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"file not found: {e.filename}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return 2


# ------------------------------------------------------------- generation

def _make_verified(text, brain, seed):
    """Returns (scene, solve_result, brain_note)."""
    if brain in ("llm-scene", "claude-scene"):
        from .brain_llm import scene_from_llm
        scene, result = scene_from_llm(text, seed)
        return scene, result, "the LLM authored the full scene"
    if brain in ("llm", "claude"):
        from .brain_llm import plan_from_llm
        plan = plan_from_llm(text)
        note = f"llm plan: {json.dumps(plan.to_json())}"
    else:
        plan = compile_text(text)
        note = f"rules plan: {json.dumps(plan.to_json())}"
    env = realize_verified(plan, seed)
    env.scene.meta["brain"] = brain
    return env.scene, env.solution, note


def _outdir(args, scene):
    d = args.out or os.path.join(RUNS_DIR, f"{scene.name[:40]}-s{scene.seed}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_artifacts(outdir, scene, result=None, showcase=None, gif=True, player="planner"):
    scene.save(os.path.join(outdir, "spec.json"))
    scene_png(scene, os.path.join(outdir, "scene.png"))
    with open(os.path.join(outdir, "ascii.txt"), "w") as f:
        f.write(ascii_frame(Engine(scene), legend=True) + "\n")
    show = showcase or result
    if show is not None:
        with open(os.path.join(outdir, "trace.json"), "w") as f:
            json.dump({"scene": scene.name, "seed": scene.seed,
                       "actions": [a.to_json() for a in show.trace]}, f)
        with open(os.path.join(outdir, "result.json"), "w") as f:
            json.dump({**show.summary(),
                       "player": player,
                       "objectives": show.engine.objective_report(),
                       "events": [[t, k, d] for t, k, d in show.engine.events]}, f, indent=2)
        replay_html(scene, show.trace, os.path.join(outdir, "replay.html"),
                    title=scene.name)
        if gif and show.trace:
            replay_gif(scene, show.trace, os.path.join(outdir, "replay.gif"))


def _print_report(scene, result, outdir, note=""):
    eng = result.engine
    print(f"\n{'=' * 62}")
    print(f"  {scene.name}   [{scene.mode}, {scene.width}x{scene.height}, seed {scene.seed}]")
    if note:
        print(f"  {note}")
    print(f"{'=' * 62}")
    print(ascii_frame(Engine(scene)))
    print()
    feats = scene.meta.get("features", {})
    short = {k: v for k, v in feats.items() if v[1] < v[0]}
    if short:
        gaps = ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in short.items())
        print(f"  note: geometry allowed fewer features than requested ({gaps})")
    for w in scene.meta.get("plan", {}).get("warnings", []):
        print(f"  note: {w}")
    attempts = scene.meta.get("verified", {}).get("attempts", [])
    if len(attempts) > 1:
        first = attempts[0][0]
        fails = ", ".join(f"s{a[0]}:{a[1]}" for a in attempts[:-1])
        print(f"  note: seed {first} failed verification; solved at seed {scene.seed} "
              f"after {len(attempts)} attempts ({fails})")
    status = "SOLVED" if result.success else f"NOT SOLVED ({result.reason})"
    print(f"  agent verdict: {status} in {result.ticks} ticks ({result.ticks / 30:.1f}s simulated)")
    for ob in eng.objective_report():
        mark = {"passed": "✓", "failed": "✗"}.get(ob["status"], "○")
        extra = f"  [{ob['progress']}]" if "progress" in ob else ""
        print(f"    {mark} {ob['description']}{extra}")
    interesting = [e for e in eng.events
                   if e[1] in ("key_collected", "door_opened", "plate_pressed", "damage")]
    if interesting:
        print("  key events: " + "; ".join(
            f"t{t} {k}({d.get('color', d.get('cause', ''))})" for t, k, d in interesting[:8]))
    shown = os.path.relpath(outdir)
    if shown.startswith(".."):
        shown = outdir  # outside the cwd: the absolute path is the readable one
    print(f"  artifacts: {shown}/  (open replay.html to watch)")


# ---------------------------------------------------------------- commands

def cmd_run(args):
    t0 = time.time()
    scene, result, note = _make_verified(args.text, args.brain, args.seed)
    showcase = None
    if args.player in ("llm", "claude"):
        from .llm_player import llm_play
        print("The LLM is playing the environment from ASCII observations...")
        showcase = llm_play(scene)
    outdir = _outdir(args, scene)
    _write_artifacts(outdir, scene, result, showcase,
                     player="llm" if showcase is not None else "planner")
    _print_report(scene, showcase or result, outdir,
                  note + f" | pipeline {time.time() - t0:.1f}s wall")
    return 0 if (showcase or result).success else 1


def cmd_gen(args):
    scene, result, note = _make_verified(args.text, args.brain, args.seed)
    outdir = _outdir(args, scene)
    _write_artifacts(outdir, scene, result)
    _print_report(scene, result, outdir, note)
    return 0


def cmd_solve(args):
    scene = Scene.load(args.spec)
    if args.player in ("llm", "claude"):
        from .llm_player import llm_play
        result = llm_play(scene)
    else:
        result = check_scene(scene)
    # never overwrite the input artifact set by default; pass --out
    # explicitly to regenerate a directory in place
    outdir = args.out or os.path.join(RUNS_DIR, f"solve-{scene.name[:40]}-s{scene.seed}")
    os.makedirs(outdir, exist_ok=True)
    _write_artifacts(outdir, scene, result,
                     player="llm" if args.player in ("llm", "claude") else "planner")
    _print_report(scene, result, outdir, f"player: {args.player}")
    return 0 if result.success else 1


def cmd_play(args):
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("play needs an interactive terminal (it opens a curses game).\n"
              "Run it directly in your shell, or watch the agent instead:\n"
              f"  python3 -m worldsmith solve {args.spec}", file=sys.stderr)
        return 2
    import curses
    from .playmode import play
    scene = Scene.load(args.spec)
    try:
        res = play(scene)
    except curses.error as e:
        print(f"terminal does not support curses ({e}); try a different "
              "terminal, or watch the agent via replay.html", file=sys.stderr)
        return 2
    print(f"result: {res}")
    return 0


def cmd_show(args):
    scene = Scene.load(args.spec)
    print(ascii_frame(Engine(scene), legend=True))
    for ob in scene.objectives:
        print(f"  - {ob.describe()}")
    return 0


def cmd_batch(args):
    """One command, N verified environments: the 'infinite' knob."""
    if args.brain in ("llm-scene", "claude-scene"):
        print("batch works with plan-producing brains (rules/llm)", file=sys.stderr)
        return 2
    if args.brain in ("llm", "claude"):
        from .brain_llm import plan_from_llm
        plan = plan_from_llm(args.text)
    else:
        plan = compile_text(args.text)
    base = args.out or os.path.join(RUNS_DIR, f"batch-{plan.name[:32]}")
    os.makedirs(base, exist_ok=True)
    rows, solved = [], 0
    t0 = time.time()
    for i in range(args.n):
        seed = args.seed + i * 1000
        try:
            env = realize_verified(plan, seed)
        except RuntimeError as e:
            rows.append((seed, "FAILED", 0, str(e)[:40]))
            continue
        scene, result = env.scene, env.solution
        d = os.path.join(base, f"env_{i:03d}_s{scene.seed}")
        os.makedirs(d, exist_ok=True)
        _write_artifacts(d, scene, result, gif=False)  # keep bulk output fast
        solved += 1
        rows.append((scene.seed, "ok", result.ticks, f"{len(env.attempts)} attempt(s)"))
        print(f"  [{i + 1}/{args.n}] seed {scene.seed}: verified in {result.ticks} ticks "
              f"({len(env.attempts)} generation attempt(s))")
    dt = time.time() - t0
    print(f"\nbatch: {solved}/{args.n} verified environments in {dt:.1f}s "
          f"({1000 * dt / max(1, args.n):.0f} ms each) -> {os.path.relpath(base)}/")
    with open(os.path.join(base, "index.json"), "w") as f:
        json.dump({"command": args.text, "plan": plan.to_json(),
                   "rows": [list(r) for r in rows]}, f, indent=2)
    return 0 if solved == args.n else 1


def cmd_verify(args):
    """Independent check of a run's core claim: replay trace.json on a fresh
    engine and compare the outcome against the recorded result.json."""
    d = args.rundir
    scene = Scene.load(os.path.join(d, "spec.json"))
    with open(os.path.join(d, "trace.json")) as f:
        trace = [Actions.from_json(a) for a in json.load(f)["actions"]]
    with open(os.path.join(d, "result.json")) as f:
        recorded = json.load(f)
    from .engine import replay
    eng = replay(scene, trace)
    checks = [
        ("success", eng.success, recorded.get("success")),
        ("ticks", eng.tick, recorded.get("ticks")),
        ("objective statuses",
         [ob.status for ob in eng.objectives],
         [o.get("status") for o in recorded.get("objectives", [])]),
        ("events", [[t_, k, d] for t_, k, d in eng.events],
         [list(e) for e in recorded.get("events", [])]),
    ]
    ok = True
    for name, got, want in checks:
        match = got == want
        ok &= match
        print(f"  {'OK ' if match else 'MISMATCH'} {name}: replayed={got!r} recorded={want!r}"
              if not match else f"  OK  {name}: {got!r}")
    if ok and recorded.get("success"):
        print("verified: trace reproduces the recorded outcome")
    elif ok:
        print("verified: trace reproduces the recorded outcome "
              "(note: the recorded episode was a FAILURE, reproduced faithfully)")
    else:
        print("VERIFICATION FAILED: replay does not match result.json")
    return 0 if ok else 1


def cmd_demo(args):
    ok = 0
    for text, seed in DEMO_PROMPTS:
        ns = argparse.Namespace(text=text, seed=seed, brain="rules", out=None)
        try:
            scene, result, note = _make_verified(text, "rules", seed)
            outdir = _outdir(ns, scene)
            _write_artifacts(outdir, scene, result)
            _print_report(scene, result, outdir, note)
            ok += result.success
        except Exception as e:
            print(f"  DEMO FAILURE for {text!r}: {e}")
    print(f"\ndemo: {ok}/{len(DEMO_PROMPTS)} prompts produced verified environments")
    return 0 if ok == len(DEMO_PROMPTS) else 1
