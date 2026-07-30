"""Generate -> prove-playable -> repair loop.

An environment only leaves this module if the solver agent has actually
beaten it inside the physics engine AND the recorded action trace replays
deterministically to the same outcome. Failed seeds are discarded and
regenerated; with procedural generation, repair is as cheap as reseeding.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import replay
from .procgen import GenPlan, build_scene
from .solver import solve, SolveResult
from .spec import Scene


@dataclass
class VerifiedEnv:
    scene: Scene
    solution: SolveResult
    attempts: list          # [(seed, outcome_str), ...]


def realize_verified(plan: GenPlan, seed: int, max_attempts: int = 10) -> VerifiedEnv:
    """Build scenes from `plan` starting at `seed` until one is proven
    playable. Raises RuntimeError if every attempt fails."""
    attempts = []
    for i in range(max_attempts):
        s = seed + i
        try:
            scene = build_scene(plan, s)
        except Exception as e:  # a bad roll of the dice; reseed
            attempts.append((s, f"generation failed: {e}"))
            continue
        outcome = check_scene(scene)
        attempts.append((s, outcome.reason))
        if outcome.success:
            scene.meta["verified"] = {
                "solver": "worldsmith-planner",
                "solution_ticks": outcome.ticks,
                "solution_seconds": round(outcome.ticks / 30.0, 2),
                "attempts": [list(a) for a in attempts],
            }
            return VerifiedEnv(scene, outcome, attempts)
    raise RuntimeError(
        "could not produce a verified-playable environment in "
        f"{max_attempts} attempts: {attempts}")


def check_scene(scene: Scene) -> SolveResult:
    """Solve a scene and confirm the trace replays deterministically."""
    result = solve(scene)
    if not result.success:
        return result
    replayed = replay(scene, result.trace)
    if not replayed.success:
        result.success = False
        result.reason = "replay_mismatch"
        result.log.append("solver won but the trace did not replay; engine nondeterminism?")
    return result
