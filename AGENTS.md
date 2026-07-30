# Agent instructions for this repo

You are likely an agent (Claude Code, Codex, or similar) whose user wants to
understand this repository. This file tells you what it is and how to give
them a good guided tour. Everything here is safe to run locally: pure Python
3.10+ stdlib, no dependencies, no network, no install step.

## What this is

WorldSmith turns plain-English commands into playable 2D game environments
(a custom physics engine, topdown and platformer modes) and includes a solver
agent that plays them. No environment is accepted unless the solver has beaten
it in the engine and its recorded inputs replay deterministically. Objectives
are code-level predicates, and the engine emits a typed event stream, so
success is checked in code rather than judged from pixels. The README explains
the design in four short paragraphs under "How it opens the problem".

## The guided tour (run these for the user, in order)

1. `python3 -m worldsmith demo`
   Seven prompts, generated, solved, and verified end to end in a few seconds.
   Show the user one of the printed reports: the ASCII map, the agent verdict,
   the objective ticks, and the key events.

2. Show a playthrough. Each `runs/<name>/` directory now contains `replay.gif`
   (animated) and `replay.html` (interactive viewer with a timeline scrubber,
   live objective status, and the event log). Open or display one for the
   user if you can; otherwise tell them which file to open in a browser.

3. `python3 -m worldsmith run "<something the user invents>"`
   Invite the user to phrase their own environment: dungeons, mazes, arenas,
   platformers; locked doors, keys, coins, crates and pressure plates, lava,
   patrolling enemies, spike pits; counts and negations both work ("3 keys",
   "no monsters"). Then walk them through the artifacts it produced.

4. `python3 -m worldsmith batch 25 "a dungeon with a key, coins and lava"`
   The scale demonstration: 25 distinct verified environments in under a
   second, each with a replayable proof of solvability.

5. `python3 -m worldsmith verify runs/<any run dir>`
   The trust-but-verify step: replays the recorded input trace on a fresh
   engine and confirms it reproduces the recorded result exactly. This is
   the submission's core claim, checkable in one command.

6. `python3 -B tests/adversarial_1.py && python3 -B tests/adversarial_2.py`
   The adversarial regression suite: hostile specs, generator-invariant
   audits, injection probes, determinism checks. All should pass.

7. The best demonstration of all: author an environment yourself. The scene
   JSON format is documented in `worldsmith/brain_llm.py` (DSL_DOC). Write
   a `spec.json` by hand (a small topdown room with a key, a door, and a goal
   is a good start), then run `python3 -m worldsmith solve your_spec.json`.
   If your scene is malformed, validation tells you exactly what is wrong; if
   it is unsolvable, the solver's verdict says so. You are exercising the
   same contract every other generation path uses.

## Cautions

- `python3 -m worldsmith play <spec>` is for humans: it opens an interactive
  curses game (headless runs exit with a short explanation rather than
  starting). Tell the user to run it themselves in their own terminal
  (WASD/arrows, space to jump, q to quit) instead of running it yourself,
  since it would block your shell waiting for keyboard input.
- `--brain llm`, `--brain llm-scene`, and `--player llm` shell out to an
  LLM CLI (the `claude` CLI by default; any model via WORLDSMITH_LLM_CMD).
  They cost real model calls and take seconds to minutes. Ask the user
  before running them; everything else in the tour is free and offline.
- `runs/` is disposable output and is gitignored. Curated artifacts live in
  `examples/`.

## After the tour: ongoing work through an agent

The CLI covers generation, verification, and rendering. For anything
programmatic (datasets, analysis, experiments), use the Python API from the
repo root; the package is not pip-installed, so imports resolve from the
working directory:

    from worldsmith import (compile_text, build_scene, realize_verified,
                            solve, replay, Scene, Engine, Actions)

    plan = compile_text("a maze with two keys")   # text -> GenPlan
    scene = build_scene(plan, seed=7)             # plan + seed -> Scene
    result = solve(scene)                         # the agent plays it
    env = realize_verified(plan, seed=7)          # reseed until proven

These are the same names the CLI uses internally; there is no hidden API.
Other things a user may ask for:

- Datasets: `batch N "<command>"` writes `runs/batch-<name>/` with an
  `index.json` manifest; each child directory holds spec/trace/result, and
  `result.json`'s event list is the programmatic reward channel.
- Extending the harness (new entity, objective, or generator): the README's
  "Extending" section maps each to the exact functions to modify.
- The determinism contract: same spec + same trace gives the same outcome on
  any machine. Changing engine constants breaks replay of previously
  recorded traces, so rerun `python3 -m worldsmith demo` (expect 7/7) after
  touching physics.
- Authoring specs: validation errors are precise and are your feedback loop;
  `verify` re-proves any run directory you produce.

## Where things live

| path | what it is |
|---|---|
| `worldsmith/spec.py` | scene schema and validation (the contract) |
| `worldsmith/engine.py` | deterministic 2D physics, events, objectives |
| `worldsmith/procgen.py` | plan + seed -> scene generators |
| `worldsmith/textgen.py` | offline text -> plan compiler |
| `worldsmith/brain_llm.py` | LLM generation backends (model-agnostic), scene DSL doc |
| `worldsmith/solver.py` | the solver agent (planner + motor control) |
| `worldsmith/verify.py` | solve, replay-check, reseed loop |
| `worldsmith/render.py` | ASCII, PNG, GIF, and HTML replay writers |
| `worldsmith/cli.py` | the commands used above |
| `examples/` | five curated environments with full artifact sets |
| `tests/` | adversarial regression suite |

Useful context when the user asks how well it works: the demo suite verifies
7/7; measured first-try solve rate is about 95% across the demo prompts and
100% after automatic reseeding; a full generate-solve-verify cycle costs 5-60 ms
depending on prompt complexity, and the solver has a 90 s wall-clock budget that reports
`budget_exhausted` honestly rather than misreporting a level as unsolvable.
