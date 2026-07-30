"""Procedural scene generation.

A GenPlan is the small parameter vector that generation brains (the rules
compiler or an LLM) produce from a text command. build_scene(plan, seed)
realizes a plan into a concrete Scene. Different seeds under the same plan
give different environments; this is the "infinite" axis.

Generators construct solvability by design (keys are always placed on the
player's side of their door, crate push lanes are checked, hazards never cut
the critical path), and every scene is additionally *proved* playable by the
solver agent afterwards (see verify.py). Belt and braces.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from .spec import Scene, EntitySpec, ObjectiveSpec, COLORS

LAYOUTS = ("dungeon", "maze", "arena", "terrain")


@dataclass
class GenPlan:
    """What to build. Produced from text by a brain, realized by build_scene."""
    name: str = "environment"
    mode: str = "topdown"            # topdown | platformer
    layout: str = "dungeon"          # dungeon | maze | arena | terrain (platformer)
    size: tuple = (24, 14)
    doors: int = 0                   # key-locked doors on the way to the goal
    plate: bool = False              # a crate-and-pressure-plate gated door
    coins: int = 0
    collect_coins: bool = False      # make collecting all coins an objective
    enemies: int = 0
    lava: str = "none"               # none | blobs | moat
    spike_pits: int = 0              # platformer only
    time_limit: float = 0.0          # 0 -> auto from size
    warnings: list = field(default_factory=list)   # honest notes, e.g. capped counts

    def to_json(self):
        d = {
            "name": self.name, "mode": self.mode, "layout": self.layout,
            "size": list(self.size), "doors": self.doors, "plate": self.plate,
            "coins": self.coins, "collect_coins": self.collect_coins,
            "enemies": self.enemies, "lava": self.lava,
            "spike_pits": self.spike_pits, "time_limit": self.time_limit,
        }
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    @staticmethod
    def from_json(d: dict) -> "GenPlan":
        p = GenPlan()
        p.name = str(d.get("name", p.name))[:64] or "environment"
        p.mode = d.get("mode", p.mode)
        p.layout = d.get("layout", p.layout)
        size = d.get("size", list(p.size))
        p.size = (int(size[0]), int(size[1]))
        p.doors = max(0, min(3, int(d.get("doors", 0))))
        p.plate = bool(d.get("plate", False))
        p.coins = max(0, min(12, int(d.get("coins", 0))))
        p.collect_coins = bool(d.get("collect_coins", False))
        p.enemies = max(0, min(6, int(d.get("enemies", 0))))
        p.lava = d.get("lava", "none")
        p.spike_pits = max(0, min(5, int(d.get("spike_pits", 0))))
        p.time_limit = float(d.get("time_limit", 0.0))
        p.warnings = [str(w) for w in d.get("warnings", [])]
        return p

    def normalize(self) -> "GenPlan":
        if self.mode not in ("topdown", "platformer"):
            self.mode = "topdown"
        if self.mode == "platformer":
            self.layout = "terrain"
            if self.plate:
                self.warnings.append(
                    "crates and pressure plates are topdown-only; dropped in platformer mode")
            if self.enemies:
                self.warnings.append(
                    "patrolling enemies are topdown-only; dropped in platformer mode")
            self.plate = False
            self.enemies = 0
        elif self.layout not in ("dungeon", "maze", "arena"):
            self.layout = "dungeon"
        if self.lava not in ("none", "blobs", "moat"):
            self.lava = "blobs" if self.lava else "none"
        W, H = self.size
        W = max(14, min(60, W))
        H = max(10, min(30, H))
        self.size = (W, H)
        return self


# ============================================================ grid helpers

def _bfs(passable, W, H, start):
    """Distance map over 4-connected passable cells. start=(c,r)."""
    dist = {start: 0}
    q = deque([start])
    while q:
        c, r = q.popleft()
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if n in dist:
                continue
            if 0 <= n[0] < W and 0 <= n[1] < H and passable(n[0], n[1]):
                dist[n] = dist[(c, r)] + 1
                q.append(n)
    return dist


def _bfs_path(passable, W, H, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        c, r = cur
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if n not in prev and 0 <= n[0] < W and 0 <= n[1] < H and passable(n[0], n[1]):
                prev[n] = cur
                q.append(n)
    return None


class GridScene:
    """Mutable tile grid + entity list while a generator works."""

    def __init__(self, plan: GenPlan, seed: int):
        self.plan = plan
        self.seed = seed
        self.rng = random.Random(seed)
        self.W, self.H = plan.size
        self.g = [["#"] * self.W for _ in range(self.H)]
        self.entities: list = []
        self.door_cells: dict = {}      # (c,r) -> color
        self.occupied: set = set()      # cells holding an entity

    def carve(self, c, r, ch="."):
        if 0 < c < self.W - 1 and 0 < r < self.H - 1:
            self.g[r][c] = ch

    def tile(self, c, r):
        if 0 <= c < self.W and 0 <= r < self.H:
            return self.g[r][c]
        return "#"

    def floor_cells(self):
        return [(c, r) for r in range(self.H) for c in range(self.W) if self.g[r][c] == "."]

    def passable(self, c, r, doors_block=True, avoid_hazard=True):
        t = self.tile(c, r)
        if t == "#":
            return False
        if avoid_hazard and t in "~^":
            return False
        if doors_block and (c, r) in self.door_cells:
            return False
        return True

    def add(self, etype, cell_or_pos, **kw):
        if isinstance(cell_or_pos[0], int):
            pos = (cell_or_pos[0] + 0.5, cell_or_pos[1] + 0.5)
            self.occupied.add(tuple(cell_or_pos))
        else:
            pos = cell_or_pos
            self.occupied.add((int(pos[0]), int(pos[1])))
        e = EntitySpec(type=etype, pos=pos, **kw)
        self.entities.append(e)
        return e

    def free_floor(self, cells=None, min_dist_from=None, min_dist=0.0):
        opts = [x for x in (cells if cells is not None else self.floor_cells())
                if x not in self.occupied and x not in self.door_cells]
        if min_dist_from is not None:
            opts = [x for x in opts
                    if (x[0] - min_dist_from[0]) ** 2 + (x[1] - min_dist_from[1]) ** 2 >= min_dist ** 2]
        return opts

    def to_scene(self, objectives, name, features=None) -> Scene:
        for (c, r), color in self.door_cells.items():
            self.entities.append(EntitySpec(type="door", pos=(c + 0.5, r + 0.5), color=color))
        meta = {"plan": self.plan.to_json(), "generator": self.plan.layout}
        if features:
            meta["features"] = features  # {name: [requested, placed]} (honest shortfall record)
        scene = Scene(
            name=name, mode=self.plan.mode, size=(self.W, self.H),
            tiles=["".join(row) for row in self.g],
            entities=self.entities, objectives=objectives, seed=self.seed,
            meta=meta,
        )
        scene.validate()
        return scene


# ========================================================= topdown layouts

def _layout_dungeon(gs: GridScene):
    """Random rooms connected by corridors. Returns list of room rects."""
    rng, W, H = gs.rng, gs.W, gs.H
    rooms = []
    target = max(3, (W * H) // 55)
    for _ in range(200):
        if len(rooms) >= target:
            break
        rw = rng.randint(4, min(8, W - 4))
        rh = rng.randint(3, min(6, H - 4))
        c0 = rng.randint(1, W - rw - 1)
        r0 = rng.randint(1, H - rh - 1)
        if any(c0 < c1 + w1 + 1 and c0 + rw + 1 > c1 and r0 < r1 + h1 + 1 and r0 + rh + 1 > r1
               for (c1, r1, w1, h1) in rooms):
            continue
        rooms.append((c0, r0, rw, rh))
        for r in range(r0, r0 + rh):
            for c in range(c0, c0 + rw):
                gs.carve(c, r)
    # connect with L-corridors, nearest-unvisited chain
    centers = [(c + w // 2, r + h // 2) for (c, r, w, h) in rooms]
    linked = [0]
    while len(linked) < len(rooms):
        best = None
        for i in linked:
            for j in range(len(rooms)):
                if j in linked:
                    continue
                d = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        _carve_corridor(gs, centers[i], centers[j])
        linked.append(j)
    # one extra loop for variety
    if len(rooms) > 3:
        i, j = rng.sample(range(len(rooms)), 2)
        _carve_corridor(gs, centers[i], centers[j])
    return rooms


def _carve_corridor(gs: GridScene, a, b):
    (c0, r0), (c1, r1) = a, b
    if gs.rng.random() < 0.5:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            gs.carve(c, r0)
        for r in range(min(r0, r1), max(r0, r1) + 1):
            gs.carve(c1, r)
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            gs.carve(c0, r)
        for c in range(min(c0, c1), max(c0, c1) + 1):
            gs.carve(c, r1)


def _layout_maze(gs: GridScene):
    """Recursive-backtracker maze on odd lattice, plus a few loops."""
    rng, W, H = gs.rng, gs.W, gs.H
    cw, ch = (W - 1) // 2, (H - 1) // 2
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))
    gs.carve(1, 1)
    while stack:
        cc, cr = stack[-1]
        opts = [(dc, dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cc + dc < cw and 0 <= cr + dr < ch and (cc + dc, cr + dr) not in visited]
        if not opts:
            stack.pop()
            continue
        dc, dr = rng.choice(opts)
        nc, nr = cc + dc, cr + dr
        gs.carve(2 * cc + 1 + dc, 2 * cr + 1 + dr)
        gs.carve(2 * nc + 1, 2 * nr + 1)
        visited.add((nc, nr))
        stack.append((nc, nr))
    # a few loops so the maze isn't a strict tree
    for _ in range(max(1, cw * ch // 30)):
        c = rng.randrange(2, W - 2)
        r = rng.randrange(2, H - 2)
        if gs.g[r][c] == "#":
            n_floor = sum(1 for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if gs.tile(c + dc, r + dr) == ".")
            if n_floor == 2:
                gs.carve(c, r)
    return []


def _layout_arena(gs: GridScene):
    rng, W, H = gs.rng, gs.W, gs.H
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            gs.carve(c, r)
    # a few pillars / short wall stubs for cover
    for _ in range((W * H) // 40):
        c = rng.randrange(3, W - 3)
        r = rng.randrange(3, H - 3)
        if rng.random() < 0.5:
            gs.g[r][c] = "#"
        else:
            for k in range(rng.randint(2, 4)):
                if rng.random() < 0.5 and c + k < W - 2:
                    gs.g[r][c + k] = "#"
                elif r + k < H - 2:
                    gs.g[r + k][c] = "#"
    return [(1, 1, W - 2, H - 2)]


def _build_topdown(plan: GenPlan, seed: int) -> Scene:
    gs = GridScene(plan, seed)
    rng = gs.rng
    rooms = {"dungeon": _layout_dungeon, "maze": _layout_maze, "arena": _layout_arena}[plan.layout](gs)

    floors = gs.floor_cells()
    if len(floors) < 30:
        raise ValueError("layout came out too small")

    # --- player and goal: far apart by BFS distance
    player_cell = rng.choice(floors)
    dist = _bfs(lambda c, r: gs.passable(c, r, doors_block=False), gs.W, gs.H, player_cell)
    reachable = [x for x in floors if x in dist]
    goal_cell = max(reachable, key=lambda x: dist[x])
    if dist[goal_cell] < max(gs.W, gs.H) // 2:  # cramped start? try the reverse
        player_cell, goal_cell = goal_cell, player_cell
    gs.add("player", player_cell)
    gs.add("goal", goal_cell)

    objectives = [ObjectiveSpec(kind="reach", target="goal", id="reach_goal")]
    colors = list(COLORS)
    rng.shuffle(colors)

    # --- key-locked doors along the critical path. The path is computed with
    # doors OPEN (so door 2..n still find a spot once door 1 blocks the way);
    # solvability comes from the key invariant: every key must be reachable
    # with ALL doors closed, so any pickup order works.
    doors_placed = 0
    for _ in range(plan.doors):
        if not colors:
            break
        path = _bfs_path(lambda c, r: gs.passable(c, r, doors_block=False),
                         gs.W, gs.H, player_cell, goal_cell)
        if not path or len(path) < 8:
            break
        # candidate door cells: corridor-like cells (2 open neighbors) mid-path
        cands = []
        for (c, r) in path[3:-3]:
            if (c, r) in gs.occupied or (c, r) in gs.door_cells:
                continue
            open_n = sum(1 for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                         if gs.tile(c + dc, r + dr) != "#" )
            if open_n == 2:
                cands.append((c, r))
        if not cands:
            break
        rng.shuffle(cands)
        placed_this_round = False
        for door_cell in cands:
            color = colors.pop()
            gs.door_cells[door_cell] = color
            # the key invariant, enforced not assumed: with ALL doors closed,
            # the new key AND every previously placed key must stay reachable
            kd = _bfs(lambda c, r: gs.passable(c, r), gs.W, gs.H, player_cell)
            key_opts = gs.free_floor(cells=[x for x in kd if kd[x] >= 3])
            prev_keys = [(int(e.pos[0]), int(e.pos[1]))
                         for e in gs.entities if e.type == "key"]
            if not key_opts or any(k not in kd for k in prev_keys):
                del gs.door_cells[door_cell]
                colors.append(color)
                continue
            weights = [kd[x] + 1 for x in key_opts]  # prefer tucked-away keys
            key_cell = rng.choices(key_opts, weights=weights, k=1)[0]
            gs.add("key", key_cell, color=color)
            doors_placed += 1
            placed_this_round = True
            break
        if not placed_this_round:
            break

    # --- crate + pressure plate gating a door
    plate_color, plate_gated = None, False
    if plan.plate and colors:
        placed = _place_plate_puzzle(gs, player_cell, goal_cell, colors)
        if placed:
            plate_color, plate_gated = placed
            objectives.append(ObjectiveSpec(kind="press", color=plate_color, id="press_plate"))

    # --- lava
    if plan.lava == "moat":
        _lava_moat(gs, goal_cell)
    elif plan.lava == "blobs":
        _lava_blobs(gs, player_cell, goal_cell)

    # --- coins (reachable ignoring doors; solver orders door-opening)
    kd = _bfs(lambda c, r: gs.passable(c, r, doors_block=False), gs.W, gs.H, player_cell)
    for _ in range(plan.coins):
        opts = gs.free_floor(cells=[x for x in kd], min_dist_from=player_cell, min_dist=2)
        if not opts:
            break
        gs.add("coin", rng.choice(opts))
    if plan.collect_coins and plan.coins:
        n = len([e for e in gs.entities if e.type == "coin"])
        objectives.append(ObjectiveSpec(kind="collect", item="coin", count=n, id="collect_coins"))

    # --- patrolling enemies on straight clear runs, away from spawn
    for _ in range(plan.enemies):
        run = _find_patrol_run(gs, player_cell, rng)
        if run is None:
            break
        (a, b) = run
        mid = ((a[0] + b[0]) / 2 + 0.5, (a[1] + b[1]) / 2 + 0.5)
        gs.add("enemy", mid,
               path=[(a[0] + 0.5, a[1] + 0.5), (b[0] + 0.5, b[1] + 0.5)],
               speed=rng.uniform(1.6, 2.6))

    objectives.append(ObjectiveSpec(kind="survive", min_hp=1, id="survive"))
    tl = plan.time_limit or (30 + 2.0 * (gs.W + gs.H) + 15 * plan.doors + 20 * plan.plate + 4 * plan.coins)
    objectives.append(ObjectiveSpec(kind="time_limit", seconds=float(tl), id="time_limit"))
    features = {
        "doors": [plan.doors, doors_placed],
        "plate": [int(plan.plate), int(bool(plate_color))],
        "plate_gated": [int(plan.plate), int(plate_gated)],
        "coins": [plan.coins, len([e for e in gs.entities if e.type == "coin"])],
        "enemies": [plan.enemies, len([e for e in gs.entities if e.type == "enemy"])],
        "lava": [int(plan.lava != "none"), int(any("~" in row for row in gs.g))],
    }
    return gs.to_scene(objectives, plan.name, features)


def _place_plate_puzzle(gs: GridScene, player_cell, goal_cell, colors):
    """Plate + crate with a clear straight push lane and standing room, plus a
    door of the plate's color gating the goal path. Placements WITHOUT a gate
    are only accepted as a last resort (the plate still carries the press
    objective, but nothing physically depends on it) and are reported as
    ungated so the caller can record the shortfall.

    Returns (color, gated) or None."""
    rng = gs.rng
    for attempt in range(400):
        opts = gs.free_floor(min_dist_from=player_cell, min_dist=2)
        if not opts:
            return None
        pc = rng.choice(opts)
        d = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        push = rng.randint(2, 3)
        crate = (pc[0] - d[0] * push, pc[1] - d[1] * push)
        stand = (crate[0] - d[0], crate[1] - d[1])
        lane = [(pc[0] - d[0] * k, pc[1] - d[1] * k) for k in range(push + 1)] + [stand]
        if not all(gs.passable(c, r) and (c, r) not in gs.occupied for (c, r) in lane):
            continue
        # crate and pushing position reachable with all current doors closed
        kd = _bfs(lambda c, r: gs.passable(c, r), gs.W, gs.H, player_cell)
        if stand not in kd or crate not in kd:
            continue
        color = colors[-1]  # reserve; only consumed on success
        # gate the goal path with a door of the plate's color
        path = _bfs_path(lambda c, r: gs.passable(c, r), gs.W, gs.H, player_cell, goal_cell)
        door_cell = None
        if path and len(path) >= 8:
            cands = [(c, r) for (c, r) in path[3:-3]
                     if (c, r) not in gs.occupied and (c, r) not in gs.door_cells
                     and (c, r) not in lane
                     and sum(1 for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                             if gs.tile(c + dc, r + dr) != "#") == 2]
            rng.shuffle(cands)
            for cand in cands:
                gs.door_cells[cand] = color
                kd2 = _bfs(lambda c, r: gs.passable(c, r), gs.W, gs.H, player_cell)
                # the door must not lock away the puzzle, and the keys placed
                # earlier must all stay reachable with every door closed
                keys_ok = all((int(e.pos[0]), int(e.pos[1])) in kd2
                              for e in gs.entities if e.type == "key")
                if stand in kd2 and crate in kd2 and pc in kd2 and keys_ok:
                    door_cell = cand
                    break
                del gs.door_cells[cand]
        if door_cell is None and attempt < 300:
            continue  # keep hunting for a placement that actually gates
        colors.pop()
        gs.add("plate", pc, color=color)
        gs.add("crate", crate)
        return color, door_cell is not None
    return None


def _lava_ok(gs: GridScene, player_cell) -> bool:
    """Lava may never break the solvability invariants: keys, crates and
    plates stay reachable with all doors CLOSED; everything else (goal,
    coins, enemies) stays reachable with doors open."""
    kd_closed = _bfs(lambda c, r: gs.passable(c, r), gs.W, gs.H, player_cell)
    kd_open = _bfs(lambda c, r: gs.passable(c, r, doors_block=False), gs.W, gs.H, player_cell)
    for e in gs.entities:
        if e.type == "player":
            continue
        cell = (int(e.pos[0]), int(e.pos[1]))
        if e.type in ("key", "crate", "plate") and cell not in kd_closed:
            return False
        if cell not in kd_open:
            return False
    return True


def _lava_moat(gs: GridScene, goal_cell):
    """Ring of lava at radius 2 around the goal with one bridge gap."""
    rng = gs.rng
    ring = []
    for dc in range(-2, 3):
        for dr in range(-2, 3):
            if max(abs(dc), abs(dr)) == 2:
                c, r = goal_cell[0] + dc, goal_cell[1] + dr
                if gs.tile(c, r) == "." and (c, r) not in gs.occupied and (c, r) not in gs.door_cells:
                    ring.append((c, r))
    if len(ring) < 4:
        return
    bridge = rng.choice(ring)
    bridge_cells = {bridge}
    # widen bridge to its ring-neighbours so the corridor is walkable
    for (c, r) in ring:
        if abs(c - bridge[0]) + abs(r - bridge[1]) == 1:
            bridge_cells.add((c, r))
            break
    placed = []
    for (c, r) in ring:
        if (c, r) not in bridge_cells:
            gs.g[r][c] = "~"
            placed.append((c, r))
    if not _lava_ok(gs, _player_cell(gs)):
        for (c, r) in placed:  # moat would strand something: revert entirely
            gs.g[r][c] = "."


def _player_cell(gs: GridScene):
    p = [e for e in gs.entities if e.type == "player"][0]
    return (int(p.pos[0]), int(p.pos[1]))


def _lava_blobs(gs: GridScene, player_cell, goal_cell):
    """Scatter lava blobs anywhere that doesn't break solvability."""
    rng = gs.rng
    n = max(2, (gs.W * gs.H) // 60)
    for _ in range(n * 4):
        if n <= 0:
            break
        opts = gs.free_floor(min_dist_from=player_cell, min_dist=3)
        if not opts:
            break
        c, r = rng.choice(opts)
        blob = [(c, r)]
        if rng.random() < 0.5:
            dc, dr = rng.choice(((1, 0), (0, 1)))
            if gs.passable(c + dc, r + dr) and (c + dc, r + dr) not in gs.occupied:
                blob.append((c + dc, r + dr))
        saved = [(bc, br, gs.g[br][bc]) for (bc, br) in blob]
        for (bc, br) in blob:
            gs.g[br][bc] = "~"
        if _lava_ok(gs, player_cell):
            n -= 1
        else:
            for (bc, br, ch) in saved:
                gs.g[br][bc] = ch


def _find_patrol_run(gs: GridScene, player_cell, rng):
    for _ in range(60):
        opts = gs.free_floor(min_dist_from=player_cell, min_dist=5)
        if not opts:
            return None
        c, r = rng.choice(opts)
        dc, dr = rng.choice(((1, 0), (0, 1)))
        length = 0
        while gs.passable(c + dc * (length + 1), r + dr * (length + 1)) and \
                (c + dc * (length + 1), r + dr * (length + 1)) not in gs.occupied:
            length += 1
            if length >= 5:
                break
        if length >= 3:
            return ((c, r), (c + dc * length, r + dr * length))
    return None


# ======================================================= platformer layout

def _build_platformer(plan: GenPlan, seed: int) -> Scene:
    gs = GridScene(plan, seed)
    rng = gs.rng
    W, H = gs.W, gs.H

    # choose spike-pit and door sites first so the terrain walk can flatten
    # around them (doors and pit rims must sit on locally flat ground)
    pits = []
    for _ in range(200):
        if len(pits) >= plan.spike_pits:
            break
        wpit = rng.randint(2, 3)
        x = rng.randint(8, max(9, W - 12))
        if all(abs(x - px) >= pw + wpit + 8 for px, pw in pits):
            pits.append((x, wpit))
    door_sites = []
    site_lo, site_hi = max(10, W // 4), W - 8
    for _ in range(500 if site_lo <= site_hi else 0):  # narrow worlds fit no door
        if len(door_sites) >= min(plan.doors, 3):
            break
        x = rng.randint(site_lo, site_hi)
        if any(px - 6 < x < px + pw + 6 for px, pw in pits):
            continue
        if any(abs(x - dx) < 8 for dx in door_sites):
            continue
        door_sites.append(x)
    flat_cols = {gx for (px, pw) in pits for gx in range(px - 3, px + pw + 4)}
    flat_cols |= {gx for dx in door_sites for gx in range(dx - 3, dx + 4)}

    # ground height as a gentle random walk (steps of at most 1, flat at pits)
    base = H - 3
    ground = [base]
    for x in range(1, W):
        step = 0 if (x in flat_cols or x < 3 or x > W - 4) else rng.choice((0, 0, 0, 1, -1))
        ground.append(max(H - 6, min(H - 2, ground[-1] + step)))

    for x in range(W):
        for r in range(0, ground[x]):
            gs.g[r][x] = "."
        for r in range(ground[x], H):
            gs.g[r][x] = "#"

    # dig the pits: floor 2-3 below the rim, spikes across the floor
    for (px, pw) in pits:
        floor_r = min(H - 1, ground[px] + rng.randint(2, 3))
        for gx in range(px, px + pw):
            for r in range(0, floor_r):
                gs.g[r][gx] = "."
            for r in range(floor_r, H):
                gs.g[r][gx] = "#"
            gs.g[floor_r - 1][gx] = "^"
            ground[gx] = floor_r

    def stand_y(x):  # y center for standing on column x
        return ground[x] - 0.5

    gs.add("player", (1.5, stand_y(1)))
    gs.add("goal", (W - 1.5, stand_y(W - 2)))
    objectives = [ObjectiveSpec(kind="reach", target="goal", id="reach_goal")]
    colors = list(COLORS)
    rng.shuffle(colors)

    # floating platforms with coins, 3 tiles above ground: jumpable
    # (apex ~3.5) while leaving 2 tiles of walking clearance underneath
    plat_xs = []
    for _ in range(plan.coins):
        for _try in range(60):
            pw = rng.randint(2, 3)
            x = rng.randint(4, W - 6 - pw)
            if any(abs(x - px) < pw + 3 for px in plat_xs):
                continue
            if any(px - 3 <= x + k < px + pwid + 3 for (px, pwid) in pits for k in range(pw)):
                continue
            if any(abs(x + k - dx) < 3 for dx in door_sites for k in range(pw)):
                continue
            # height set from the columns BESIDE the platform: the player
            # jumps up from a side, never from underneath it
            py = min(ground[x - 1], ground[x + pw]) - 3
            if py < 3:
                continue
            # platform row plus 2 rows of headroom must be clear air
            if any(gs.tile(x + k, py - dr) != "." for k in range(pw) for dr in (0, 1, 2)):
                continue
            for k in range(pw):
                gs.g[py][x + k] = "#"
            gs.add("coin", (x + pw / 2.0, py - 0.5))
            plat_xs.append(x)
            break
    if plan.collect_coins and plan.coins:
        n = len([e for e in gs.entities if e.type == "coin"])
        if n:
            objectives.append(ObjectiveSpec(kind="collect", item="coin", count=n, id="collect_coins"))

    # locked door barriers on the pre-flattened sites: a wall pillar too tall
    # to jump, with the door tile at ground level. The key is always placed on
    # walkable ground LEFT of its door (the player's side), so ordering is
    # solvable by construction even with several doors.
    doors_placed = 0
    for x in sorted(door_sites):
        if not colors:
            break
        gr = ground[x]
        if gr - 5 < 1:
            continue
        color = colors.pop()
        for r in range(gr - 5, gr - 1):
            gs.g[r][x] = "#"
        gs.door_cells[(x, gr - 1)] = color
        key_opts = [kx for kx in range(2, x - 1)
                    if gs.tile(kx, ground[kx] - 1) == "."
                    and (kx, ground[kx] - 1) not in gs.occupied
                    and not any(px <= kx < px + pw for px, pw in pits)
                    and abs(kx - x) > 1
                    and not any(abs(kx - dx) < 2 for dx in door_sites)]
        if not key_opts:
            del gs.door_cells[(x, gr - 1)]
            for r in range(gr - 5, gr - 1):
                gs.g[r][x] = "."
            colors.append(color)
            continue
        kx = rng.choice(key_opts)
        gs.add("key", (kx + 0.5, ground[kx] - 0.5), color=color)
        doors_placed += 1

    objectives.append(ObjectiveSpec(kind="survive", min_hp=1, id="survive"))
    tl = plan.time_limit or (30 + 2.5 * W + 15 * doors_placed)
    objectives.append(ObjectiveSpec(kind="time_limit", seconds=float(tl), id="time_limit"))
    features = {
        "doors": [plan.doors, doors_placed],
        "coins": [plan.coins, len([e for e in gs.entities if e.type == "coin"])],
        "spike_pits": [plan.spike_pits, len(pits)],
    }
    return gs.to_scene(objectives, plan.name, features)


# ================================================================== entry

def build_scene(plan: GenPlan, seed: int) -> Scene:
    plan = plan.normalize()
    if plan.mode == "platformer":
        return _build_platformer(plan, seed)
    return _build_topdown(plan, seed)
