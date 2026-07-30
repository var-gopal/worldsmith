"""LLM-as-policy: let any LLM play an environment from ASCII observations.

This is the 2D stand-in for a vision policy: instead of rendered frames it
observes ASCII frames + a state summary, and instead of controller hardware
it emits short action programs ("R 20, D 10"). Purely a demonstration that
the environments are playable by an LLM without the planner; the planner in
solver.py remains the workhorse for verification.
"""

from __future__ import annotations

import re
import subprocess

from .brain_llm import ask_llm, LLMUnavailable
from .engine import Engine, Actions
from .render import ascii_frame
from .solver import SolveResult
from .spec import Scene

PLAY_PROMPT = """You are playing a 2D {mode} game. Map legend:
# wall  . floor  ~ lava(hurts)  ^ spikes(hurt)  P you  G goal  o coin  k key
D closed door (touch it while holding its key to open)  / open door
C pushable crate  _ pressure plate (push crate onto it)  E enemy (hurts)

{mode_help}
Objectives (all must pass): {objectives}

Current state: {state}

Map:
{frame}

Reply with ONLY a short action program: comma-separated steps `<KEYS> <ticks>`.
KEYS is a combo of U D L R (movement{jump_note}); 30 ticks = 1 second.
Example: `R 25, RU 10, U 20`. Maximum 6 steps. No prose."""


def llm_play(scene: Scene, max_queries: int = 40, verbose=print) -> SolveResult:
    eng = Engine(scene)
    trace, log = [], []
    mode_help = ("Top-down view; U/D/L/R move you up/down/left/right." if scene.mode == "topdown"
                 else "Side view with gravity; L/R walk, J jumps (about 3 tiles high, 4-5 across).")
    for q in range(max_queries):
        if eng.done:
            break
        prompt = PLAY_PROMPT.format(
            mode=scene.mode, mode_help=mode_help,
            objectives="; ".join(ob.spec.describe() for ob in eng.objectives),
            state=eng.state_summary(), frame=ascii_frame(eng),
            jump_note=", J jump" if scene.mode == "platformer" else "")
        try:
            reply = ask_llm(prompt, timeout=300)
        except (subprocess.TimeoutExpired, LLMUnavailable) as e:
            log.append(f"[q{q}] LLM call failed ({type(e).__name__}); stopping early")
            verbose(f"  LLM call failed ({type(e).__name__}); stopping early")
            break
        program = _parse_program(reply)
        log.append(f"[q{q}] {reply.strip()[:120]!r} -> {len(program)} steps")
        verbose(f"  LLM move {q + 1}: {reply.strip()[:80]}")
        if not program:
            program = [(Actions(), 5)]
        for act, ticks in program:
            for _ in range(min(ticks, 90)):
                if eng.done:
                    break
                eng.step(act)
                trace.append(act)
    reason = "solved" if eng.success else ("query_budget" if not eng.done else "failed")
    return SolveResult(eng.success, trace, len(trace), reason, log, eng)


def _parse_program(reply: str):
    out = []
    for part in reply.split(",")[:6]:
        m = re.search(r"\b([UDLRJ]{1,3})\s*[: ]\s*(\d{1,3})", part.strip(), re.IGNORECASE)
        if not m:
            continue
        keys = m.group(1).upper()
        out.append((Actions("L" in keys, "R" in keys, "U" in keys,
                            "D" in keys, "J" in keys), int(m.group(2))))
    return out
