"""A small deterministic 2D physics engine with two modes.

- topdown:    no gravity; the player steers in 4/8 directions with
              acceleration and friction (Zelda-like). Crates are pushable.
- platformer: gravity, jumping, ground friction and air control (Mario-like).

Design goals, in order: determinism (fixed timestep, no runtime randomness,
snapshot/restore), verifiability (every game event is logged with its tick and
objectives are evaluated in code), and speed (pure Python, axis-separated AABB
collision against a static tile grid, fast enough to run thousands of steps
per second, which the solver relies on).

Coordinates: x grows right, y grows DOWN (row 0 is the top). Tile (c, r)
occupies the square [c, c+1] x [r, r+1]. Gravity is +y.

Actions mirror a controller: left/right/up/down booleans plus jump. This is
deliberately the same shape as a vision-policy action space (WASD + discrete
buttons) so any policy that emits controller actions can drive the player.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .spec import Scene, MAX_HP

DT = 1.0 / 30.0  # fixed timestep

# --- topdown tuning ---
TD_ACCEL = 55.0
TD_MAX_V = 5.5
TD_FRICTION = 60.0

# --- platformer tuning ---
PF_GRAVITY = 30.0
PF_JUMP_V = 14.5          # apex ~3.5 tiles, airtime ~0.97 s
PF_MOVE_ACCEL = 55.0
PF_AIR_ACCEL = 30.0
PF_MAX_VX = 6.0
PF_MAX_FALL = 16.0
PF_FRICTION = 50.0

PLAYER_HP = MAX_HP
INVULN_S = 1.0
HAZARD_SHRINK = 0.18      # hazard tiles hurt only near their center
EPS = 1e-6


@dataclass
class Actions:
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False
    jump: bool = False

    def to_json(self):
        # compact: string of pressed keys
        s = ""
        for flag, ch in ((self.left, "L"), (self.right, "R"), (self.up, "U"),
                         (self.down, "D"), (self.jump, "J")):
            if flag:
                s += ch
        return s

    @staticmethod
    def from_json(s: str) -> "Actions":
        return Actions("L" in s, "R" in s, "U" in s, "D" in s, "J" in s)


@dataclass
class Body:
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0

    def aabb(self, pad: float = 0.0):
        hw, hh = self.w / 2 + pad, self.h / 2 + pad
        return (self.x - hw, self.y - hh, self.x + hw, self.y + hh)

    def overlaps(self, other: "Body", pad: float = 0.0) -> bool:
        ax0, ay0, ax1, ay1 = self.aabb(pad)
        bx0, by0, bx1, by1 = other.aabb()
        return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


@dataclass
class Item:          # coin or key
    id: str
    kind: str        # "coin" | "key"
    x: float
    y: float
    color: str = ""
    taken: bool = False


@dataclass
class Door:
    id: str
    color: str
    c: int
    r: int
    open: bool = False


@dataclass
class Plate:
    id: str
    color: str
    x: float
    y: float
    latched: bool = False


@dataclass
class Enemy:
    id: str
    body: Body
    path: list                    # [(x, y), ...] patrolled ping-pong
    speed: float
    leg: int = 0                  # index of waypoint we are moving toward
    fwd: int = 1                  # +1 forward along path, -1 backward


@dataclass
class Objective:
    spec: object                  # ObjectiveSpec
    status: str = "pending"       # pending | passed | failed
    progress: int = 0             # for collect counts


class Engine:
    """Simulates one Scene. Deterministic given the action sequence."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.mode = scene.mode
        self.W, self.H = scene.size
        # static grid
        self.wall = [[ch == "#" for ch in row] for row in scene.tiles]
        self.hazard = [[ch in "~^" for ch in row] for row in scene.tiles]

        p = scene.entities_of("player")[0]
        ph = 0.9 if self.mode == "platformer" else 0.7
        self.player = Body(p.pos[0], p.pos[1], 0.7, ph)
        self.hp = PLAYER_HP
        self.invuln = 0.0
        self.last_safe = (self.player.x, self.player.y)
        self.grounded = False
        self.inventory: dict = {}                 # color -> key count
        self.keys_collected: dict = {}            # color -> total ever collected

        self.goals = [Body(e.pos[0], e.pos[1], 0.9, 0.9) for e in scene.entities_of("goal")]
        self.goal_ids = [e.id for e in scene.entities_of("goal")]
        self.items = [Item(e.id, e.type, e.pos[0], e.pos[1], e.color or "")
                      for e in scene.entities if e.type in ("coin", "key")]
        self.doors = [Door(e.id, e.color, int(e.pos[0]), int(e.pos[1]))
                      for e in scene.entities_of("door")]
        self.plates = [Plate(e.id, e.color, e.pos[0], e.pos[1])
                       for e in scene.entities_of("plate")]
        self.crates = [Body(e.pos[0], e.pos[1], 0.85, 0.85)
                       for e in scene.entities_of("crate")]
        self.crate_ids = [e.id for e in scene.entities_of("crate")]
        self.enemies = [Enemy(e.id, Body(e.pos[0], e.pos[1], 0.8, 0.8),
                              list(e.path or [tuple(e.pos)]), e.speed)
                        for e in scene.entities_of("enemy")]

        self.objectives = [Objective(o) for o in scene.objectives]
        self.time_limit = 120.0
        for o in scene.objectives:
            if o.kind == "time_limit":
                self.time_limit = o.seconds

        self.tick = 0
        self.events: list = []                    # (tick, kind, data)
        self._goal_overlap: set = set()
        self.done = False
        self.success = False

    # ------------------------------------------------------------ queries
    def solid(self, c: int, r: int) -> bool:
        if c < 0 or r < 0 or c >= self.W or r >= self.H:
            return True
        if self.wall[r][c]:
            return True
        for d in self.doors:
            if not d.open and d.c == c and d.r == r:
                return True
        return False

    def hazard_at(self, c: int, r: int) -> bool:
        return 0 <= c < self.W and 0 <= r < self.H and self.hazard[r][c]

    @property
    def time(self) -> float:
        return self.tick * DT

    def emit(self, kind: str, **data):
        self.events.append((self.tick, kind, data))

    # ------------------------------------------------------ tile collision
    def _move_axis(self, b: Body, delta: float, axis: str) -> bool:
        """Move body along one axis, clamping against solid tiles.
        Returns True if a collision clamped the movement."""
        if delta == 0:
            return False
        hit = False
        if axis == "x":
            b.x += delta
        else:
            b.y += delta
        x0, y0, x1, y1 = b.aabb()
        c0, c1 = int(math.floor(x0)), int(math.ceil(x1) - 1)
        r0, r1 = int(math.floor(y0)), int(math.ceil(y1) - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not self.solid(c, r):
                    continue
                # overlap with tile [c,c+1]x[r,r+1]?
                if x0 < c + 1 and x1 > c and y0 < r + 1 and y1 > r:
                    hit = True
                    if axis == "x":
                        if delta > 0:
                            b.x = c - b.w / 2 - EPS
                        else:
                            b.x = c + 1 + b.w / 2 + EPS
                        b.vx = 0.0
                    else:
                        if delta > 0:
                            b.y = r - b.h / 2 - EPS
                        else:
                            b.y = r + 1 + b.h / 2 + EPS
                        b.vy = 0.0
                    x0, y0, x1, y1 = b.aabb()
        return hit

    def _body_blocked(self, b: Body) -> bool:
        """Is body b overlapping any solid tile right now?"""
        x0, y0, x1, y1 = b.aabb(-EPS)
        for r in range(int(math.floor(y0)), int(math.ceil(y1))):
            for c in range(int(math.floor(x0)), int(math.ceil(x1))):
                if self.solid(c, r):
                    return True
        return False

    def _push_crates(self, mover: Body, delta: float, axis: str):
        """After mover moved along axis, shove any overlapped crate. If the
        crate can't move the full amount, the mover is clamped against it."""
        for i, crate in enumerate(self.crates):
            if not mover.overlaps(crate):
                continue
            # penetration depth along axis
            if axis == "x":
                pen = (mover.aabb()[2] - crate.aabb()[0]) if delta > 0 else (mover.aabb()[0] - crate.aabb()[2])
            else:
                pen = (mover.aabb()[3] - crate.aabb()[1]) if delta > 0 else (mover.aabb()[1] - crate.aabb()[3])
            if abs(pen) < EPS:
                continue
            self._move_crate(i, pen, axis)
            # snap-assist: pushed crates drift toward the lane center line,
            # which keeps continuous pushes aligned to the tile grid
            if axis == "x":
                lane = math.floor(crate.y) + 0.5
                crate.y += max(-2.5 * DT, min(2.5 * DT, lane - crate.y))
            else:
                lane = math.floor(crate.x) + 0.5
                crate.x += max(-2.5 * DT, min(2.5 * DT, lane - crate.x))
            # re-clamp mover against the crate's final position
            if mover.overlaps(crate):
                if axis == "x":
                    if delta > 0:
                        mover.x = crate.aabb()[0] - mover.w / 2 - EPS
                    else:
                        mover.x = crate.aabb()[2] + mover.w / 2 + EPS
                    mover.vx = 0.0
                else:
                    if delta > 0:
                        mover.y = crate.aabb()[1] - mover.h / 2 - EPS
                    else:
                        mover.y = crate.aabb()[3] + mover.h / 2 + EPS
                    mover.vy = 0.0

    def _move_crate(self, idx: int, delta: float, axis: str):
        """Move crate idx along axis with tile + other-crate collision."""
        crate = self.crates[idx]
        self._move_axis(crate, delta, axis)
        # crates block each other: undo overlap against others
        for j, other in enumerate(self.crates):
            if j == idx or not crate.overlaps(other):
                continue
            if axis == "x":
                if delta > 0:
                    crate.x = other.aabb()[0] - crate.w / 2 - EPS
                else:
                    crate.x = other.aabb()[2] + crate.w / 2 + EPS
            else:
                if delta > 0:
                    crate.y = other.aabb()[1] - crate.h / 2 - EPS
                else:
                    crate.y = other.aabb()[3] + crate.h / 2 + EPS
        return True

    # --------------------------------------------------------------- step
    def step(self, act: Actions):
        if self.done:
            return
        self.tick += 1
        if self.invuln > 0:
            self.invuln = max(0.0, self.invuln - DT)

        if self.mode == "topdown":
            self._step_topdown(act)
        else:
            self._step_platformer(act)

        self._update_enemies()
        self._sense()
        self._check_hazards()
        if not self.done:  # a fatal hit already ended the episode: no
            self._update_objectives()  # objective events after episode_end
        if self.time > self.time_limit and not self.done:
            self._finish(False, "time_limit")

    def _step_topdown(self, act: Actions):
        p = self.player
        ax = (act.right - act.left) * TD_ACCEL
        ay = (act.down - act.up) * TD_ACCEL
        if ax == 0:
            p.vx = _toward(p.vx, 0.0, TD_FRICTION * DT)
        else:
            p.vx += ax * DT
        if ay == 0:
            p.vy = _toward(p.vy, 0.0, TD_FRICTION * DT)
        else:
            p.vy += ay * DT
        speed = math.hypot(p.vx, p.vy)
        if speed > TD_MAX_V:
            p.vx *= TD_MAX_V / speed
            p.vy *= TD_MAX_V / speed
        dx, dy = p.vx * DT, p.vy * DT
        self._move_axis(p, dx, "x")
        self._push_crates(p, dx, "x")
        self._move_axis(p, dy, "y")
        self._push_crates(p, dy, "y")

    def _step_platformer(self, act: Actions):
        p = self.player
        # grounded: a solid tile just below the feet
        self.grounded = self._on_ground(p)
        accel = PF_MOVE_ACCEL if self.grounded else PF_AIR_ACCEL
        move = act.right - act.left
        if move == 0:
            if self.grounded:
                p.vx = _toward(p.vx, 0.0, PF_FRICTION * DT)
        else:
            p.vx += move * accel * DT
        p.vx = max(-PF_MAX_VX, min(PF_MAX_VX, p.vx))
        if (act.jump or act.up) and self.grounded:
            p.vy = -PF_JUMP_V
            self.grounded = False
            self.emit("jump")
        p.vy = min(PF_MAX_FALL, p.vy + PF_GRAVITY * DT)
        self._move_axis(p, p.vx * DT, "x")
        self._move_axis(p, p.vy * DT, "y")

    def _on_ground(self, b: Body) -> bool:
        x0, _, x1, y1 = b.aabb()
        r = int(math.floor(y1 + 0.05))
        if abs(y1 - r) >= 0.08:   # feet not resting at a tile boundary
            return False
        for c in range(int(math.floor(x0 + 0.05)), int(math.ceil(x1 - 0.05))):
            if self.solid(c, r):
                return True
        return False

    def _update_enemies(self):
        for e in self.enemies:
            if len(e.path) < 2:
                continue
            tx, ty = e.path[e.leg]
            dx, dy = tx - e.body.x, ty - e.body.y
            dist = math.hypot(dx, dy)
            step = max(e.speed, 0.0) * DT
            if dist <= step or dist < 1e-9:  # never divide by a zero distance
                e.body.x, e.body.y = tx, ty
                nxt = e.leg + e.fwd
                if nxt < 0 or nxt >= len(e.path):
                    e.fwd = -e.fwd
                    nxt = e.leg + e.fwd
                e.leg = nxt
            else:
                e.body.x += dx / dist * step
                e.body.y += dy / dist * step

    def _sense(self):
        p = self.player
        # items
        for it in self.items:
            if it.taken:
                continue
            if abs(p.x - it.x) < 0.65 and abs(p.y - it.y) < 0.65:
                it.taken = True
                if it.kind == "key":
                    self.inventory[it.color] = self.inventory.get(it.color, 0) + 1
                    self.keys_collected[it.color] = self.keys_collected.get(it.color, 0) + 1
                    self.emit("key_collected", id=it.id, color=it.color)
                else:
                    self.emit("coin_collected", id=it.id)
        # doors: touch with matching key -> consume key, open
        for d in self.doors:
            if d.open:
                continue
            db = Body(d.c + 0.5, d.r + 0.5, 1.0, 1.0)
            if p.overlaps(db, pad=0.22) and self.inventory.get(d.color, 0) > 0:
                self.inventory[d.color] -= 1
                d.open = True
                self.emit("door_opened", id=d.id, color=d.color, by="key")
        # plates: latch only when a CRATE rests on them (the player's weight
        # is not enough; the press objective certifies the manipulation)
        for pl in self.plates:
            if pl.latched:
                continue
            pressed = any(abs(crate.x - pl.x) < 0.35 and abs(crate.y - pl.y) < 0.35
                          for crate in self.crates)
            if pressed:
                pl.latched = True
                self.emit("plate_pressed", id=pl.id, color=pl.color)
                for d in self.doors:
                    if d.color == pl.color and not d.open:
                        d.open = True
                        self.emit("door_opened", id=d.id, color=d.color, by="plate")
        # goal (rising-edge: one event per arrival, not per overlapping tick)
        for gid, g in zip(self.goal_ids, self.goals):
            if p.overlaps(g):
                if gid not in self._goal_overlap:
                    self._goal_overlap.add(gid)
                    self.emit("goal_touched", id=gid)
            else:
                self._goal_overlap.discard(gid)

    def _check_hazards(self):
        p = self.player
        hurt = False
        x0, y0, x1, y1 = p.aabb(-HAZARD_SHRINK)
        for r in range(int(math.floor(y0)), int(math.ceil(y1))):
            for c in range(int(math.floor(x0)), int(math.ceil(x1))):
                if self.hazard_at(c, r):
                    hurt = True
        enemy_hit = None
        for e in self.enemies:
            if p.overlaps(e.body, pad=-0.08):
                enemy_hit = e.id
        if (hurt or enemy_hit) and self.invuln <= 0:
            self.hp -= 1
            self.invuln = INVULN_S
            self.emit("damage", cause=(enemy_hit or "hazard"), hp=self.hp)
            # respawn at last safe spot (Zelda-style) so the sim can't wedge
            p.x, p.y = self.last_safe
            p.vx = p.vy = 0.0
            if self.hp <= 0:
                self.emit("death")
                self._finish(False, "death")
                return
        if not hurt and not enemy_hit:
            safe_here = True
            if self.mode == "platformer" and not self._on_ground(p):
                safe_here = False
            for e in self.enemies:
                if p.overlaps(e.body, pad=0.8):
                    safe_here = False
            if safe_here:
                self.last_safe = (p.x, p.y)

    def _update_objectives(self):
        all_tasks_passed = True
        for ob in self.objectives:
            o = ob.spec
            if ob.status == "failed":
                if o.kind not in ("survive", "time_limit"):
                    all_tasks_passed = False
                continue
            if o.kind == "reach":
                tgt = o.target or "goal"
                if tgt == "goal" or tgt in self.goal_ids:
                    touched = any(k == "goal_touched" and (tgt == "goal" or d.get("id") == tgt)
                                  for _, k, d in self.events)
                else:  # reach an arbitrary entity: position proximity
                    # (0.9: wide enough to trigger while standing against a
                    # solid target such as a crate, which blocks overlap)
                    e = self.scene.entity_by_id(tgt)
                    touched = e is not None and abs(self.player.x - e.pos[0]) < 0.9 \
                        and abs(self.player.y - e.pos[1]) < 0.9
                if touched:
                    if ob.status != "passed":
                        ob.status = "passed"
                        self.emit("objective_passed", id=o.id)
                else:
                    all_tasks_passed = False
            elif o.kind == "collect":
                base = o.item.split(":")[0]
                color = o.item.split(":")[1] if ":" in o.item else None
                n = sum(1 for it in self.items
                        if it.taken and it.kind == base and (color is None or it.color == color))
                ob.progress = n
                if n >= o.count:
                    if ob.status != "passed":
                        ob.status = "passed"
                        self.emit("objective_passed", id=o.id)
                else:
                    all_tasks_passed = False
            elif o.kind == "open":
                ok = any(d.open and (o.color is None or d.color == o.color) for d in self.doors)
                if ok:
                    if ob.status != "passed":
                        ob.status = "passed"
                        self.emit("objective_passed", id=o.id)
                else:
                    all_tasks_passed = False
            elif o.kind == "press":
                ok = any(pl.latched and (o.color is None or pl.color == o.color) for pl in self.plates)
                if ok:
                    if ob.status != "passed":
                        ob.status = "passed"
                        self.emit("objective_passed", id=o.id)
                else:
                    all_tasks_passed = False
            elif o.kind == "survive":
                if self.hp < o.min_hp:
                    ob.status = "failed"
            elif o.kind == "time_limit":
                ob.status = "passed" if self.time <= o.seconds else "failed"
        if all_tasks_passed and not self.done:
            constraints_ok = all(ob.status != "failed" for ob in self.objectives)
            for ob in self.objectives:
                if ob.spec.kind in ("survive", "time_limit") and ob.status == "pending":
                    ob.status = "passed"
            self._finish(constraints_ok, "objectives_complete")

    def _finish(self, success: bool, reason: str):
        self.done = True
        self.success = success
        self.emit("episode_end", success=success, reason=reason)

    # ---------------------------------------------------------- lifecycle
    def snapshot(self):
        return copy.deepcopy((
            self.player, self.hp, self.invuln, self.last_safe, self.grounded,
            self.inventory, self.keys_collected, self.items, self.doors,
            self.plates, self.crates, self.enemies, self.objectives,
            self.tick, self.events, self.done, self.success, self._goal_overlap,
        ))

    def restore(self, snap):
        (self.player, self.hp, self.invuln, self.last_safe, self.grounded,
         self.inventory, self.keys_collected, self.items, self.doors,
         self.plates, self.crates, self.enemies, self.objectives,
         self.tick, self.events, self.done, self.success,
         self._goal_overlap) = copy.deepcopy(snap)

    # ------------------------------------------------------------ reports
    def objective_report(self) -> list:
        out = []
        for ob in self.objectives:
            entry = {"id": ob.spec.id, "kind": ob.spec.kind,
                     "description": ob.spec.describe(), "status": ob.status}
            if ob.spec.kind == "collect":
                entry["progress"] = f"{ob.progress}/{ob.spec.count}"
            out.append(entry)
        return out

    def state_summary(self) -> dict:
        return {
            "tick": self.tick,
            "time_s": round(self.time, 2),
            "player": [round(self.player.x, 2), round(self.player.y, 2)],
            "hp": self.hp,
            "keys": {k: v for k, v in self.inventory.items() if v},
            "doors_open": [d.id for d in self.doors if d.open],
            "objectives": self.objective_report(),
        }


def _toward(v: float, target: float, amount: float) -> float:
    if v < target:
        return min(target, v + amount)
    return max(target, v - amount)


def replay(scene: Scene, trace: list) -> Engine:
    """Re-run a recorded action trace on a fresh engine (verification)."""
    eng = Engine(scene)
    for act in trace:
        if eng.done:
            break
        eng.step(act)
    return eng
