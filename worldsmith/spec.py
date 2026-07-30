"""Scene specification: the JSON contract between generation and simulation.

A Scene is the single source of truth for an environment. Everything else
(the physics engine, the solver agent, the renderers) consumes a validated
Scene. Generation backends (rules, procgen, LLM) all produce Scenes.

Tile legend (one string per row, row 0 = top):
    '#'  wall (solid)
    '.'  floor / empty air
    '~'  lava   (hazard, non-solid)
    '^'  spikes (hazard, non-solid)

Entity types: player, goal, coin, key, door, crate, plate, enemy.
Doors and keys are matched by color; a plate latches open all doors of its
color. Objectives are verifiable predicates evaluated inside the engine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

MODES = ("topdown", "platformer")
MAX_HP = 3  # player hit points; survive objectives cannot demand more
TILE_CHARS = {"#", ".", "~", "^"}
ENTITY_TYPES = ("player", "goal", "coin", "key", "door", "crate", "plate", "enemy")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
OBJECTIVE_KINDS = ("reach", "collect", "open", "press", "survive", "time_limit")


class SpecError(ValueError):
    """Raised when a scene spec is malformed, with a human-readable reason."""


@dataclass
class EntitySpec:
    type: str
    pos: tuple  # (x, y) center in world units; tile (c, r) spans [c,c+1]x[r,r+1]
    id: str = ""
    color: Optional[str] = None          # key / door / plate
    path: Optional[list] = None          # enemy patrol waypoints [[x,y], ...]
    speed: float = 2.0                   # enemy patrol speed (units/s)

    def to_json(self):
        d = {"type": self.type, "pos": [self.pos[0], self.pos[1]], "id": self.id}
        if self.color is not None:
            d["color"] = self.color
        if self.path is not None:
            d["path"] = [list(p) for p in self.path]
            d["speed"] = self.speed
        return d


@dataclass
class ObjectiveSpec:
    kind: str
    id: str = ""
    target: Optional[str] = None   # reach: entity id or "goal"
    item: Optional[str] = None     # collect: "coin" | "key:red"
    count: int = 1                 # collect: how many
    color: Optional[str] = None    # open / press: door or plate color
    min_hp: int = 1                # survive
    seconds: float = 120.0         # time_limit

    def to_json(self):
        d = {"kind": self.kind, "id": self.id}
        if self.kind == "reach":
            d["target"] = self.target or "goal"
        elif self.kind == "collect":
            d["item"] = self.item
            d["count"] = self.count
        elif self.kind in ("open", "press"):
            d["color"] = self.color
        elif self.kind == "survive":
            d["min_hp"] = self.min_hp
        elif self.kind == "time_limit":
            d["seconds"] = self.seconds
        return d

    def describe(self) -> str:
        if self.kind == "reach":
            return f"reach the {self.target or 'goal'}"
        if self.kind == "collect":
            noun = (self.item or "item").replace(":", " ")
            return f"collect {self.count} {noun}{'s' if self.count > 1 and ':' not in (self.item or '') else ''}"
        if self.kind == "open":
            return f"open the {self.color} door" if self.color else "open a door"
        if self.kind == "press":
            return f"press the {self.color} pressure plate (push a crate onto it)"
        if self.kind == "survive":
            return f"finish with at least {self.min_hp} hp"
        if self.kind == "time_limit":
            return f"finish within {self.seconds:.0f}s"
        return self.kind


@dataclass
class Scene:
    name: str
    mode: str
    size: tuple                    # (W, H) in tiles
    tiles: list                    # list[str], H rows of W chars
    entities: list = field(default_factory=list)     # list[EntitySpec]
    objectives: list = field(default_factory=list)   # list[ObjectiveSpec]
    seed: int = 0
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def tile(self, c: int, r: int) -> str:
        if 0 <= c < self.width and 0 <= r < self.height:
            return self.tiles[r][c]
        return "#"

    def entities_of(self, etype: str) -> list:
        return [e for e in self.entities if e.type == etype]

    def entity_by_id(self, eid: str) -> Optional[EntitySpec]:
        for e in self.entities:
            if e.id == eid:
                return e
        return None

    # --------------------------------------------------------------- json
    def to_json(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "seed": self.seed,
            "size": [self.width, self.height],
            "tiles": list(self.tiles),
            "entities": [e.to_json() for e in self.entities],
            "objectives": [o.to_json() for o in self.objectives],
            "meta": self.meta,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @staticmethod
    def from_json(d: dict) -> "Scene":
        """Parse a scene dict. Structural junk of any shape raises SpecError
        with a readable message (never a bare TypeError/AttributeError), so
        generation feedback loops can always relay what was wrong."""
        try:
            size = d.get("size", [0, 0])
            scene = Scene(
                name=str(d.get("name", "unnamed")),
                mode=str(d.get("mode", "topdown")),
                size=(int(float(size[0])), int(float(size[1]))),
                tiles=[str(row) for row in d.get("tiles", [])],
                seed=int(d.get("seed", 0)),
                meta=dict(d.get("meta", {})),
            )
            for i, ed in enumerate(d.get("entities", [])):
                if not isinstance(ed, dict):
                    raise SpecError(f"entities[{i}] must be an object, got {type(ed).__name__}")
                pos = ed.get("pos", (0, 0))
                scene.entities.append(EntitySpec(
                    type=str(ed.get("type", "")),
                    pos=(float(pos[0]), float(pos[1])),
                    id=str(ed.get("id", "")) or f"{ed.get('type', 'e')}_{i}",
                    color=str(ed["color"]) if ed.get("color") is not None else None,
                    path=[(float(p[0]), float(p[1])) for p in ed["path"]] if ed.get("path") else None,
                    speed=float(ed.get("speed", 2.0)),
                ))
            for i, od in enumerate(d.get("objectives", [])):
                if not isinstance(od, dict):
                    raise SpecError(f"objectives[{i}] must be an object, got {type(od).__name__}")
                scene.objectives.append(ObjectiveSpec(
                    kind=str(od.get("kind", "")),
                    id=str(od.get("id", "")) or f"obj_{i}",
                    target=str(od["target"]) if od.get("target") is not None else None,
                    item=str(od["item"]) if od.get("item") is not None else None,
                    count=int(od.get("count", 1)),
                    color=str(od["color"]) if od.get("color") is not None else None,
                    min_hp=int(od.get("min_hp", 1)),
                    seconds=float(od.get("seconds", 120.0)),
                ))
        except SpecError:
            raise
        except Exception as e:
            raise SpecError(f"malformed scene JSON ({type(e).__name__}: {e})")
        return scene

    @staticmethod
    def load(path: str) -> "Scene":
        with open(path) as f:
            d = json.load(f)
        scene = Scene.from_json(d)
        scene.validate()
        return scene

    # --------------------------------------------------------- validation
    def validate(self):
        """Raise SpecError with a precise message on the first problem found."""
        if self.mode not in MODES:
            raise SpecError(f"mode must be one of {MODES}, got {self.mode!r}")
        W, H = self.size
        if not (4 <= W <= 200 and 4 <= H <= 200):
            raise SpecError(f"size must be between 4x4 and 200x200, got {W}x{H}")
        if len(self.tiles) != H:
            raise SpecError(f"tiles has {len(self.tiles)} rows, size says {H}")
        for r, row in enumerate(self.tiles):
            if len(row) != W:
                raise SpecError(f"tiles row {r} has {len(row)} chars, size says {W}")
            bad = set(row) - TILE_CHARS
            if bad:
                raise SpecError(f"tiles row {r} contains unknown chars {sorted(bad)}")

        # assign ids to any entity that lacks one
        counts: dict = {}
        for e in self.entities:
            if e.type not in ENTITY_TYPES:
                raise SpecError(f"unknown entity type {e.type!r}")
            counts[e.type] = counts.get(e.type, 0) + 1
            if not e.id:
                e.id = f"{e.type}_{counts[e.type]}"
        ids = [e.id for e in self.entities]
        if len(ids) != len(set(ids)):
            raise SpecError("duplicate entity ids")

        players = self.entities_of("player")
        if len(players) != 1:
            raise SpecError(f"scene needs exactly 1 player, got {len(players)}")

        def tile_of(e):
            # floor, NOT int(): int() truncates toward zero, which would let
            # small negative coordinates masquerade as tile 0
            return (int(math.floor(e.pos[0])), int(math.floor(e.pos[1])))

        # doors first: their tiles block enemy patrols and host nothing else
        door_tiles = set()
        for e in self.entities_of("door"):
            if (tile_of(e)) in door_tiles:
                raise SpecError(f"two doors share tile {tile_of(e)}")
            door_tiles.add(tile_of(e))

        player_tile = None
        for e in self.entities:
            x, y = e.pos
            if not (math.isfinite(x) and math.isfinite(y)):
                raise SpecError(f"{e.id} position {e.pos} is not finite")
            c, r = tile_of(e)
            if not (0 <= c < W and 0 <= r < H):
                raise SpecError(f"{e.id} at {e.pos} occupies tile ({c},{r}), outside the {W}x{H} grid")
            if e.type != "door" and self.tile(c, r) == "#":
                raise SpecError(f"{e.id} at {e.pos} is inside a wall tile")
            if e.type == "player":
                player_tile = (c, r)
            if self.mode == "platformer" and e.type in ("crate", "plate", "enemy"):
                raise SpecError(f"{e.id}: {e.type} entities are not supported in platformer mode")
            if e.type in ("key", "door", "plate"):
                if e.color not in COLORS:
                    raise SpecError(f"{e.id} needs a color from {COLORS}, got {e.color!r}")
            if e.type == "enemy":
                if not (math.isfinite(e.speed) and 0 < e.speed <= 20):
                    raise SpecError(f"{e.id} speed must be finite and in (0, 20], got {e.speed!r}")
                if e.path:
                    pts = list(e.path)
                    for px, py in pts:
                        if not (math.isfinite(px) and math.isfinite(py)):
                            raise SpecError(f"{e.id} patrol point ({px},{py}) is not finite")
                        if not (0 <= px <= W and 0 <= py <= H):
                            raise SpecError(f"{e.id} patrol point ({px},{py}) out of bounds")
                    # patrol legs must not cross walls or door tiles
                    # (enemies have no collision and doors may be closed)
                    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                        steps = max(1, int(max(abs(bx - ax), abs(by - ay)) * 4))
                        for k in range(steps + 1):
                            sx, sy = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
                            sc, sr = int(math.floor(sx)), int(math.floor(sy))
                            if self.tile(sc, sr) == "#" or (sc, sr) in door_tiles:
                                raise SpecError(
                                    f"{e.id} patrol leg ({ax},{ay})->({bx},{by}) passes through "
                                    f"a wall or door at tile ({sc},{sr})")

        # nothing may sit on the player's spawn tile or on a door tile
        for e in self.entities:
            c, r = tile_of(e)
            if e.type in ("door", "crate") and (c, r) == player_tile:
                raise SpecError(f"{e.id} sits on the player's spawn tile ({c},{r})")
            if e.type != "door" and (c, r) in door_tiles:
                raise SpecError(f"{e.id} sits on a door tile ({c},{r})")

        # doors need SUFFICIENT openers: keys are consumed one per door, so
        # n doors of a color need n keys, or a same-color plate (opens all
        # doors of its color) with at least one crate to press it
        has_crate = bool(self.entities_of("crate"))
        for color in {d.color for d in self.entities_of("door")}:
            n_doors = len([d for d in self.entities_of("door") if d.color == color])
            n_keys = len([k for k in self.entities_of("key") if k.color == color])
            plated = has_crate and any(p.color == color for p in self.entities_of("plate"))
            if not plated and n_keys < n_doors:
                raise SpecError(
                    f"{n_doors} {color} door(s) but only {n_keys} {color} key(s); keys are "
                    f"consumed per door; add keys or a {color} plate plus a crate")
        if self.entities_of("plate") and not has_crate:
            raise SpecError("scene has pressure plates but no crate to press them")

        obj_ids = [o.id for o in self.objectives]
        if len(obj_ids) != len(set(obj_ids)):
            raise SpecError("duplicate objective ids")
        for o in self.objectives:
            if o.kind not in OBJECTIVE_KINDS:
                raise SpecError(f"unknown objective kind {o.kind!r}")
            if o.kind == "collect" and not (1 <= o.count <= 999):
                raise SpecError(f"collect count must be 1..999, got {o.count}")
            if o.kind == "time_limit" and not (math.isfinite(o.seconds) and o.seconds > 0):
                raise SpecError(f"time_limit must be positive and finite, got {o.seconds}")
            if o.kind == "survive" and not (1 <= o.min_hp <= MAX_HP):
                raise SpecError(f"survive min_hp must be 1..{MAX_HP} (max hp), got {o.min_hp}")
            if o.kind == "press":
                if not self.entities_of("crate"):
                    raise SpecError("press objective needs at least one crate in the scene")
                if not [p for p in self.entities_of("plate")
                        if o.color is None or p.color == o.color]:
                    raise SpecError(f"press objective: no matching plate in scene")
            if o.kind == "open" and not [d for d in self.entities_of("door")
                                         if o.color is None or d.color == o.color]:
                raise SpecError("open objective: no matching door in scene")
            if o.kind == "reach":
                tgt = o.target or "goal"
                if tgt != "goal" and self.entity_by_id(tgt) is None:
                    raise SpecError(f"reach objective targets unknown entity {tgt!r}")
                if tgt == "goal" and not self.entities_of("goal"):
                    raise SpecError("reach-goal objective but scene has no goal entity")
            if o.kind == "collect":
                if not o.item:
                    raise SpecError("collect objective needs an item ('coin' or 'key:<color>')")
                base = o.item.split(":")[0]
                if base not in ("coin", "key"):
                    raise SpecError(f"collect item must be coin or key:<color>, got {o.item!r}")
                have = len([e for e in self.entities
                            if e.type == base and (":" not in o.item or e.color == o.item.split(":")[1])])
                if have < o.count:
                    raise SpecError(f"collect objective wants {o.count} x {o.item}, scene has {have}")
        if not any(o.kind not in ("survive", "time_limit") for o in self.objectives):
            raise SpecError("scene needs at least one task objective (reach/collect/open/press)")
        return self
