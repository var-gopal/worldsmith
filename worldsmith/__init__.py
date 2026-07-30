"""WorldSmith: text commands -> verified-playable 2D game environments.

Public surface:
    from worldsmith import Scene, Engine, Actions, solve, build_scene, compile_text
"""

from .spec import Scene, EntitySpec, ObjectiveSpec, SpecError
from .engine import Engine, Actions, replay, DT
from .procgen import GenPlan, build_scene
from .textgen import compile_text
from .solver import solve, SolveResult
from .verify import realize_verified, check_scene, VerifiedEnv

__all__ = [
    "Scene", "EntitySpec", "ObjectiveSpec", "SpecError",
    "Engine", "Actions", "replay", "DT",
    "GenPlan", "build_scene", "compile_text",
    "solve", "SolveResult",
    "realize_verified", "check_scene", "VerifiedEnv",
]
