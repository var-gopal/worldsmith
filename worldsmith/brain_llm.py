"""LLM generation backends: real language understanding via any LLM CLI.

The backend is model-agnostic. By default prompts go to the `claude` CLI;
set WORLDSMITH_LLM_CMD to route them to any other command-line LLM. The
command is run through the shell; if it contains "{prompt}" that placeholder
is replaced (shell-quoted), otherwise the prompt is piped to stdin. Examples:
    WORLDSMITH_LLM_CMD='claude -p --output-format text'         (the default)
    WORLDSMITH_LLM_CMD='codex exec --skip-git-repo-check -s read-only -'


Two levels of trust, both funneled through the same validation + verification
pipeline as the rules brain, so a hallucinated spec can never escape:

  plan mode (default): the LLM translates the command into a GenPlan; the
      procgen realizes it. Robust: the model only picks parameters.
  scene mode: the LLM authors the full Scene JSON (tiles included). Creative:
      arbitrary layouts. Schema errors and unsolvable scenes are fed back to
      the model for repair, a bounded number of times.

Everything else in WorldSmith works offline with no LLM at all.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess

from .procgen import GenPlan
from .spec import Scene, SpecError
from .verify import check_scene

PLAN_PROMPT = """You are the front-end of a 2D game environment generator.
Translate the user's command into a generation plan. Reply with ONLY a JSON
object, no prose, no code fences.

Fields (all optional, defaults shown):
  "name": short-kebab-case-name
  "mode": "topdown" (Zelda-like dungeon physics) | "platformer" (gravity + jumping)
  "layout": "dungeon" | "maze" | "arena"    (topdown only; platformer is terrain)
  "size": [width, height]                   (tiles; 14..60 x 10..30)
  "doors": 0..3       locked doors on the way to the goal (keys spawn matched)
  "plate": false      true -> a crate must be pushed onto a pressure plate to open a door
  "coins": 0..12      collectible coins
  "collect_coins": false   true -> collecting all coins is a required objective
  "enemies": 0..6     patrolling enemies that damage on contact (topdown only)
  "lava": "none" | "blobs" | "moat"          (moat = ring of lava near the goal)
  "spike_pits": 0..5  spike pits to jump over (platformer only)
  "time_limit": 0     seconds; 0 = auto

Interpret the spirit of the command: "hard" means more hazards/enemies,
"peaceful" means none, "huge maze" means layout maze + large size, etc.

User command: {command}"""

SCENE_PROMPT = """You are a 2D game level designer. Author a complete scene as
JSON for the engine described below. Reply with ONLY the JSON object.

{dsl}

Design brief: {command}

Rules of thumb: keep corridors at least 1 tile wide; the player must be able
to reach every key before its door; in platformer mode the player can jump
about 3 tiles high and 4-5 tiles across; place the goal far from the player.
"""

DSL_DOC = """Scene JSON schema:
{
  "name": "kebab-name", "mode": "topdown"|"platformer", "seed": 0,
  "size": [W, H],                      // 14..60 x 10..30
  "tiles": ["#####...", ...],          // H strings of W chars; row 0 = top
                                       // '#' wall, '.' floor/air, '~' lava, '^' spikes
  "entities": [                        // pos = [x, y] center, tile (c,r) center is [c+0.5, r+0.5]
    {"type":"player","pos":[x,y]},     // exactly one
    {"type":"goal","pos":[x,y]},
    {"type":"coin","pos":[x,y]},
    {"type":"key","color":"red","pos":[x,y]},     // colors: red blue green yellow purple orange
    {"type":"door","color":"red","pos":[c+0.5,r+0.5]},  // solid until opened by key or plate of same color
    {"type":"crate","pos":[x,y]},                 // pushable (topdown)
    {"type":"plate","color":"blue","pos":[x,y]},  // a CRATE resting on it opens same-color doors
                                                  // (player weight does not press plates)
    {"type":"enemy","pos":[x,y],"path":[[x1,y1],[x2,y2]],"speed":2.0}
                                                  // speed in (0, 20]; patrol legs must not
                                                  // cross walls or door tiles
  ],
  // crates, plates and enemies are topdown-only (rejected in platformer mode)
  "objectives": [                      // verifiable predicates, checked in-engine
    {"kind":"reach","target":"goal"},
    {"kind":"collect","item":"coin","count":3},   // or "key:red"
    {"kind":"open","color":"red"},
    {"kind":"press","color":"blue"},
    {"kind":"survive","min_hp":1},
    {"kind":"time_limit","seconds":90}
  ]
}
In platformer mode gravity pulls +y (downward on screen); the player needs
solid ground ('#') under spawn, keys, coins and goal."""


class LLMUnavailable(RuntimeError):
    pass


def ask_llm(prompt: str, timeout: int = 240) -> str:
    """Send a prompt to the configured LLM CLI and return its text reply."""
    cmd = os.environ.get("WORLDSMITH_LLM_CMD", "").strip()
    if cmd:
        if "{prompt}" in cmd:
            shell_cmd = cmd.replace("{prompt}", shlex.quote(prompt))
            stdin = None
        else:
            shell_cmd = cmd
            stdin = prompt
        r = subprocess.run(shell_cmd, shell=True, input=stdin,
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise LLMUnavailable(f"WORLDSMITH_LLM_CMD failed: {r.stderr.strip()[:400]}")
        return r.stdout.strip()
    exe = shutil.which("claude")
    if not exe:
        raise LLMUnavailable(
            "no LLM backend: `claude` CLI not on PATH and WORLDSMITH_LLM_CMD unset; "
            "use --brain rules, or point WORLDSMITH_LLM_CMD at any LLM CLI")
    r = subprocess.run([exe, "-p", prompt, "--output-format", "text"],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise LLMUnavailable(f"claude CLI failed: {r.stderr.strip()[:400]}")
    return r.stdout.strip()


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in LLM reply: {text[:200]!r}")
    return json.loads(m.group(0))


def plan_from_llm(command: str) -> GenPlan:
    reply = ask_llm(PLAN_PROMPT.format(command=command))
    return GenPlan.from_json(_extract_json(reply)).normalize()


def scene_from_llm(command: str, seed: int, max_repairs: int = 3):
    """Full scene authoring with a validate/verify feedback-repair loop.
    Returns (scene, solve_result)."""
    prompt = SCENE_PROMPT.format(dsl=DSL_DOC, command=command)
    feedback = ""
    last_err = "unknown"
    for round_no in range(1 + max_repairs):
        reply = ask_llm(prompt + feedback)
        try:
            scene = Scene.from_json(_extract_json(reply))
            scene.seed = seed
            scene.validate()
        except Exception as e:  # ANY malformed reply becomes repair feedback
            last_err = f"spec invalid: {e}"
            feedback = f"\n\nYour previous attempt failed validation: {e}\nFix it and resend the full JSON."
            continue
        result = check_scene(scene)
        if result.success:
            scene.meta["brain"] = "llm-scene"
            scene.meta["verified"] = {
                "solver": "worldsmith-planner",
                "solution_ticks": result.ticks,
                "authored_by": "llm-scene",
                "repair_rounds": round_no,
            }
            return scene, result
        last_err = f"not playable: solver got stuck ({result.reason})"
        feedback = ("\n\nYour previous scene validated but the solver agent could not "
                    f"beat it (reason: {result.reason}). Likely an unreachable key/goal or "
                    "an impossible jump. Adjust the layout and resend the full JSON.")
    raise RuntimeError(f"LLM scene authoring failed after {max_repairs} repairs: {last_err}")
