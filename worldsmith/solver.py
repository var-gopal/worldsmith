"""The solver agent: plays a Scene in the real engine and records its actions.

This is deliberate two-level control, mirroring how a trained policy would be
deployed:

  * a LOGICAL planner decides the next subgoal (grab that key, open that
    door, push the crate onto the plate, collect that coin, reach the goal),
    re-planning as the world changes;
  * a MOTOR controller turns each subgoal into controller-style actions
    (left/right/up/down/jump) executed tick by tick in the physics engine:
    grid-BFS waypoint following in topdown, and a jump graph built by
    *simulating the engine itself* in platformer mode.

Nothing teleports and nothing cheats: verification of an environment is the
solver actually beating it, and the recorded action trace replays
deterministically (see verify.py).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from .engine import Engine, Actions, DT
from .spec import Scene

MAX_WALL_TICKS = 30 * 240      # absolute safety cap on any solve (sim ticks)
DEFAULT_DEADLINE_S = 90.0      # wall-clock budget for planning + execution


@dataclass
class SolveResult:
    success: bool
    trace: list                 # list[Actions]
    ticks: int
    reason: str
    log: list = field(default_factory=list)
    engine: object = None

    def summary(self) -> dict:
        return {
            "success": self.success,
            "ticks": self.ticks,
            "sim_seconds": round(self.ticks * DT, 2),
            "reason": self.reason,
            "log": self.log,
        }


def solve(scene: Scene, deadline_s: float = DEFAULT_DEADLINE_S) -> SolveResult:
    cls = _TopdownControl if scene.mode == "topdown" else _PlatformerControl
    return cls(scene, deadline_s).run()


# =============================================================== base class

class _Control:
    def __init__(self, scene: Scene, deadline_s: float = DEFAULT_DEADLINE_S):
        self.scene = scene
        self.eng = Engine(scene)
        self.trace: list = []
        self.log: list = []
        self.deadline = time.monotonic() + deadline_s

    # -- primitive: advance one tick
    def act(self, a: Actions = None):
        a = a or Actions()
        self.eng.step(a)
        self.trace.append(a)

    def out_of_budget(self) -> bool:
        return len(self.trace) >= MAX_WALL_TICKS or time.monotonic() > self.deadline

    def player_tile(self):
        return (int(self.eng.player.x), int(self.eng.player.y))

    def note(self, msg: str):
        self.log.append(f"[t={self.eng.tick}] {msg}")

    # ---------------------------------------------------- logical planner
    def run(self) -> SolveResult:
        eng = self.eng
        for _ in range(80):
            if eng.done or self.out_of_budget():
                break
            progressed = False
            for kind, target in self._candidates():
                if eng.done:
                    progressed = True
                    break
                ok = self._attempt(kind, target)
                if ok or eng.done:  # an "unsuccessful" move can still end the episode
                    progressed = True
                    break
            if not progressed:
                if self.out_of_budget():
                    # planning aborted on the wall clock, NOT proven stuck;
                    # a bigger budget might well solve this scene
                    self.note("wall-clock budget exhausted during planning")
                    return SolveResult(False, self.trace, len(self.trace),
                                       "budget_exhausted", self.log, eng)
                self.note("no achievable subgoal remains; giving up")
                return SolveResult(False, self.trace, len(self.trace),
                                   "stuck", self.log, eng)
        # let objective bookkeeping settle
        for _ in range(3):
            if eng.done:
                break
            self.act()
        reason = "solved" if eng.success else (
            "budget_exhausted" if self.out_of_budget() else
            next((d.get("reason", "ended") for _, k, d in reversed(eng.events)
                  if k == "episode_end"), "incomplete"))
        return SolveResult(eng.success, self.trace, len(self.trace),
                           reason, self.log, eng)

    def _pending(self, *kinds):
        return [ob for ob in self.eng.objectives
                if ob.status == "pending" and ob.spec.kind in kinds]

    def _candidates(self):
        """Ordered (kind, target) subgoals given current world state."""
        eng = self.eng
        out = []
        closed = [d for d in eng.doors if not d.open]
        plate_colors = {p.color for p in eng.plates}
        # 1. keys we still need: to unlock closed doors (beyond what we hold),
        #    or to satisfy pending collect-key objectives
        need = {}
        for color in {d.color for d in closed} - plate_colors:
            need[color] = sum(1 for d in closed if d.color == color) \
                - eng.inventory.get(color, 0)
        any_key_need = 0
        for ob in self._pending("collect"):
            item = ob.spec.item or ""
            if not item.startswith("key"):
                continue
            if ":" in item:
                color = item.split(":")[1]
                taken = sum(1 for it in eng.items
                            if it.taken and it.kind == "key" and it.color == color)
                need[color] = max(need.get(color, 0), ob.spec.count - taken)
            else:
                any_key_need = max(any_key_need, ob.spec.count - ob.progress)
        keys = [it for it in eng.items if it.kind == "key" and not it.taken
                and (need.get(it.color, 0) > 0 or any_key_need > 0)]
        keys.sort(key=lambda it: self._dist_to((it.x, it.y)))
        for it in keys:
            out.append(("item", it))
        # 2. doors we can open right now
        for d in sorted(closed, key=lambda d: self._dist_to((d.c + 0.5, d.r + 0.5))):
            if eng.inventory.get(d.color, 0) > 0:
                out.append(("door", d))
        # 3. crate -> plate (press objectives, or plate-gated closed doors)
        need_plates = {ob.spec.color for ob in self._pending("press")}
        for d in closed:
            if d.color in plate_colors and eng.inventory.get(d.color, 0) == 0:
                need_plates.add(d.color)
        for pl in eng.plates:
            if not pl.latched and (pl.color in need_plates or None in need_plates):
                out.append(("plate", pl))
        # 4. coins if a collect objective still needs them
        for ob in self._pending("collect"):
            if ob.spec.item and ob.spec.item.startswith("coin"):
                coins = [it for it in eng.items if it.kind == "coin" and not it.taken]
                coins.sort(key=lambda it: self._dist_to((it.x, it.y)))
                for it in coins[: max(0, ob.spec.count - ob.progress)]:
                    out.append(("item", it))
        # 5. reach objectives last
        for ob in self._pending("reach"):
            tgt = ob.spec.target or "goal"
            if tgt == "goal":
                for gid, g in zip(self.eng.goal_ids, self.eng.goals):
                    out.append(("reach", (g.x, g.y)))
            else:
                e = self.scene.entity_by_id(tgt)
                if e:
                    out.append(("reach", (e.pos[0], e.pos[1])))
        return out

    def _attempt(self, kind, target) -> bool:
        if kind == "item":
            label = f"{target.kind}{':' + target.color if target.color else ''} {target.id}"
            ok = self.goto((target.x, target.y), tol=0.45)
            ok = ok and target.taken
            self.note(f"{'collected' if ok else 'failed to reach'} {label}")
            return ok
        if kind == "door":
            ok = self.open_door(target)
            self.note(f"{'opened' if ok else 'failed to open'} door {target.id} ({target.color})")
            return ok
        if kind == "plate":
            ok = self.press_plate(target)
            self.note(f"{'pressed' if ok else 'failed to press'} plate {target.id} ({target.color})")
            return ok
        if kind == "reach":
            ok = self.goto(target, tol=0.55)
            self.note(f"{'reached' if ok else 'failed to reach'} target at {target}")
            return ok
        return False

    def _dist_to(self, pos) -> float:
        return math.hypot(pos[0] - self.eng.player.x, pos[1] - self.eng.player.y)

    # subclasses provide: goto, open_door, press_plate


# ============================================================ topdown motor

class _TopdownControl(_Control):

    # -- grid passability for walking (crates and closed doors block)
    def _passable(self, c, r, ignore_crates=False):
        eng = self.eng
        if eng.solid(c, r) or eng.hazard_at(c, r):
            return False
        if not ignore_crates:
            for crate in eng.crates:
                if int(crate.x) == c and int(crate.y) == r:
                    return False
        return True

    def _bfs_path(self, start, goal, ignore_crates=False):
        W, H = self.eng.W, self.eng.H
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
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cur[0] + dc, cur[1] + dr)
                if n not in prev and 0 <= n[0] < W and 0 <= n[1] < H \
                        and self._passable(n[0], n[1], ignore_crates):
                    prev[n] = cur
                    q.append(n)
        return None

    def goto(self, pos, tol=0.4, max_ticks=3000) -> bool:
        """Walk to a world position along a BFS path, waiting out enemies."""
        eng = self.eng
        goal_tile = (int(pos[0]), int(pos[1]))
        spent = 0
        while spent < max_ticks and not eng.done and not self.out_of_budget():
            if self._dist_to(pos) <= tol:
                return True
            path = self._bfs_path(self.player_tile(), goal_tile)
            if path is None:
                return False
            waypoints = [(c + 0.5, r + 0.5) for (c, r) in path[1:]] + [pos]
            spent += self._follow(waypoints, pos, tol, max_ticks - spent)
        return self._dist_to(pos) <= tol

    def _follow(self, waypoints, final_pos, tol, budget) -> int:
        """Follow waypoints; returns ticks spent. Exits early to trigger a
        replan when stuck (pushed off course, knocked back, path changed)."""
        eng = self.eng
        spent = 0
        wp_i = 0
        best = self._dist_to(final_pos)
        since_progress = 0
        waited = 0
        while spent < budget and wp_i < len(waypoints) and not eng.done:
            wx, wy = waypoints[wp_i]
            dx, dy = wx - eng.player.x, wy - eng.player.y
            if abs(dx) < 0.18 and abs(dy) < 0.18:
                wp_i += 1
                continue
            if self._dist_to(final_pos) <= tol:
                return spent
            # enemy caution: wait for a patrol to clear the way ahead
            if waited < 240 and self._enemy_blocks(wx, wy):
                self.act(Actions())
                spent += 1
                waited += 1
                continue
            a = Actions(left=dx < -0.1, right=dx > 0.1, up=dy < -0.1, down=dy > 0.1)
            self.act(a)
            spent += 1
            d = self._dist_to(final_pos)
            if d < best - 0.05:
                best = d
                since_progress = 0
            else:
                since_progress += 1
                if since_progress > 60:
                    return spent  # stuck -> replan
        return spent

    def _enemy_blocks(self, wx, wy) -> bool:
        p = self.eng.player
        for e in self.eng.enemies:
            ex, ey = e.body.x, e.body.y
            d_p = math.hypot(ex - p.x, ey - p.y)
            if d_p < 1.1:
                return False  # too close to stand still; keep moving
            # distance from enemy to the segment player -> waypoint
            seg = _pt_seg_dist(ex, ey, p.x, p.y, wx, wy)
            if d_p < 2.6 and seg < 1.15:
                return True
        return False

    def open_door(self, door) -> bool:
        """Stand next to the door and lean into it; the key sensor opens it."""
        eng = self.eng
        sides = [(door.c + dc, door.r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        sides = [s for s in sides if self._passable(*s)]
        sides.sort(key=lambda s: self._dist_to((s[0] + 0.5, s[1] + 0.5)))
        for s in sides:
            if not self.goto((s[0] + 0.5, s[1] + 0.5), tol=0.3):
                continue
            for _ in range(30):
                if door.open or eng.done:
                    return door.open
                dx = door.c + 0.5 - eng.player.x
                dy = door.r + 0.5 - eng.player.y
                self.act(Actions(left=dx < -0.05, right=dx > 0.05,
                                 up=dy < -0.05, down=dy > 0.05))
            if door.open:
                return True
        return door.open

    # ------------------------------------------------------ crate pushing
    def press_plate(self, plate) -> bool:
        eng = self.eng
        target = (int(plate.x), int(plate.y))
        # nearest crate with a sokoban plan
        plans = []
        for idx, crate in enumerate(eng.crates):
            plan = self._sokoban(idx, target)
            if plan:
                plans.append((len(plan), idx, plan))
        if not plans:
            return False
        _, idx, plan = min(plans)
        for attempt in range(3):
            ok = self._execute_pushes(idx, plan)
            if plate.latched or eng.done:
                return plate.latched
            plan = self._sokoban(idx, target)
            if not plan:
                return False
        return plate.latched

    def _sokoban(self, crate_idx, target):
        """BFS over (crate_cell, pusher reachability). Returns push list
        [(dir, dest_cell), ...] or None."""
        eng = self.eng
        W, H = eng.W, eng.H
        crate0 = (int(eng.crates[crate_idx].x), int(eng.crates[crate_idx].y))
        item_cells = {(int(it.x), int(it.y)) for it in eng.items if not it.taken}
        other_crates = {(int(c.x), int(c.y)) for j, c in enumerate(eng.crates) if j != crate_idx}

        def free(c, r):
            return not eng.solid(c, r) and not eng.hazard_at(c, r) and (c, r) not in other_crates

        def player_reach(crate, start):
            seen = {start}
            q = deque([start])
            while q:
                cur = q.popleft()
                for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (cur[0] + dc, cur[1] + dr)
                    if n not in seen and 0 <= n[0] < W and 0 <= n[1] < H \
                            and n != crate and free(*n):
                        seen.add(n)
                        q.append(n)
            return seen

        start_p = self.player_tile()
        prev = {(crate0, _region_key(player_reach(crate0, start_p))): (None, None, start_p)}
        q = deque([(crate0, start_p)])
        expansions = 0
        max_expansions = max(4000, min(40000, W * H * 6))
        while q and expansions < max_expansions:
            # every expansion runs a full-grid BFS below, so poll the clock
            # each time; a coarse stride overshoots the deadline by seconds
            if time.monotonic() > self.deadline:
                return None
            crate, ppos = q.popleft()
            expansions += 1
            reach = player_reach(crate, ppos)
            rkey = _region_key(reach)
            if crate == target:
                # rebuild push sequence
                pushes = []
                cur = (crate, rkey)
                while prev[cur][0] is not None:
                    parent, push, parent_p = prev[cur]
                    pushes.append(push)
                    cur = parent
                return pushes[::-1]
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                stand = (crate[0] - dc, crate[1] - dr)
                dest = (crate[0] + dc, crate[1] + dr)
                if stand not in reach:
                    continue
                if not free(*dest) or dest in item_cells:
                    continue
                nstate_p = crate  # player ends where crate was
                nreach = player_reach(dest, nstate_p)
                nkey = (dest, _region_key(nreach))
                if nkey in prev:
                    continue
                prev[nkey] = ((crate, rkey), ((dc, dr), dest), nstate_p)
                q.append((dest, nstate_p))
        return None

    def _execute_pushes(self, crate_idx, pushes) -> bool:
        eng = self.eng
        crate = eng.crates[crate_idx]
        for (dc, dr), dest in pushes:
            stand = (int(crate.x) - dc, int(crate.y) - dr)
            if not self.goto((stand[0] + 0.5, stand[1] + 0.5), tol=0.22):
                return False
            tx, ty = dest[0] + 0.5, dest[1] + 0.5
            for _ in range(140):
                if eng.done:
                    return False
                along = (crate.x - tx) * dc + (crate.y - ty) * dr
                if along > -0.10:
                    break
                self.act(Actions(left=dc < 0, right=dc > 0, up=dr < 0, down=dr > 0))
            else:
                return False
            for _ in range(4):
                self.act(Actions())
        return True


def _region_key(reach: set):
    return min(reach) if reach else None


def _pt_seg_dist(px, py, ax, ay, bx, by) -> float:
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


# ========================================================= platformer motor

# canned jump scripts; each is a list of Actions replayed tick-by-tick
def _macro_scripts():
    out = {}
    for name, dx in (("L", -1), ("R", 1)):
        d = dict(left=dx < 0, right=dx > 0)
        out[f"jump_{name}_full"] = [Actions(jump=True, **d)] + [Actions(**d)] * 44
        out[f"jump_{name}_short"] = [Actions(jump=True, **d)] + [Actions(**d)] * 7 + [Actions()] * 36
        out[f"jump_{name}_up"] = [Actions(jump=True)] * 4 + [Actions(**d)] * 12 + [Actions()] * 30
        out[f"fall_{name}"] = [Actions(**d)] * 50
    return out


MACROS = _macro_scripts()


class _PlatformerControl(_Control):

    def __init__(self, scene: Scene, deadline_s: float = DEFAULT_DEADLINE_S):
        super().__init__(scene, deadline_s)
        self._graph_cache = None
        self._graph_doors = None

    # ------------------------------------------------------- standing map
    def _standing_tiles(self):
        eng = self.eng
        tiles = set()
        for r in range(eng.H - 1):
            for c in range(eng.W):
                if not eng.solid(c, r) and not eng.hazard_at(c, r) and eng.solid(c, r + 1):
                    tiles.add((c, r))
        return tiles

    def _graph(self):
        """Movement graph over standing tiles. Jump/fall edges are found by
        SIMULATING the actual engine from a standing start, so every edge is
        a physically-proven maneuver. Cached per door configuration."""
        doorstate = tuple(sorted((d.id, d.open) for d in self.eng.doors))
        if self._graph_cache is not None and self._graph_doors == doorstate:
            return self._graph_cache
        eng = self.eng
        stand = self._standing_tiles()
        edges = {t: {} for t in stand}
        # walk edges between same-row neighbours
        for (c, r) in stand:
            for dc in (-1, 1):
                if (c + dc, r) in stand:
                    edges[(c, r)][(c + dc, r)] = ("walk", 8)
        # simulated jump / fall edges
        sandbox = Engine(self.scene)
        for d_main, d_sb in zip(self.eng.doors, sandbox.doors):
            d_sb.open = d_main.open
        base = sandbox.snapshot()
        for (c, r) in stand:
            if time.monotonic() > self.deadline:
                break  # budget: ship the partial graph; run() reports budget_exhausted
            for mname, script in MACROS.items():
                sandbox.restore(base)
                sandbox.done = False
                p = sandbox.player
                p.x, p.y = c + 0.5, r + 1 - p.h / 2 - 1e-6
                p.vx = p.vy = 0.0
                landing = self._simulate_macro(sandbox, script)
                if landing is None or landing == (c, r):
                    continue
                if landing not in stand:
                    continue
                cost = len(script) + 8
                cur = edges[(c, r)].get(landing)
                if cur is None or cost < cur[1]:
                    edges[(c, r)][landing] = (mname, cost)
        self._graph_cache = edges
        self._graph_doors = doorstate
        return edges

    @staticmethod
    def _simulate_macro(sandbox: Engine, script) -> tuple | None:
        hp0 = sandbox.hp
        airborne = 0
        for i in range(len(script) + 40):
            a = script[i] if i < len(script) else Actions()
            sandbox.step(a)
            if sandbox.hp < hp0 or sandbox.done:
                return None
            if not sandbox._on_ground(sandbox.player):
                airborne += 1
            elif airborne >= 3:
                return (int(sandbox.player.x), int(sandbox.player.y))
        return None

    # -------------------------------------------------------------- motor
    def _settle(self) -> bool:
        """Wait until grounded and nearly still (start of every maneuver)."""
        for _ in range(90):
            if self.eng.done:
                return False
            if self.eng._on_ground(self.eng.player) and abs(self.eng.player.vx) < 0.15:
                return True
            self.act(Actions())
        return False

    def _walk_to_x(self, x, tol=0.12, max_ticks=90) -> bool:
        eng = self.eng
        for _ in range(max_ticks):
            if eng.done:
                return False
            p = eng.player
            dx = x - p.x
            if abs(dx) <= tol:
                if abs(p.vx) < 0.8:
                    return True
                self.act(Actions())        # close: brake with ground friction
                continue
            # coast if current speed would overshoot within ~3 ticks
            if p.vx * (1 if dx > 0 else -1) * 0.1 > abs(dx):
                self.act(Actions())
            else:
                self.act(Actions(left=dx < 0, right=dx > 0))
        return abs(x - eng.player.x) <= tol * 2.5

    def _do_edge(self, src, dst, kind) -> bool:
        eng = self.eng
        if kind == "walk":
            ok = self._walk_to_x(dst[0] + 0.5, tol=0.1, max_ticks=40)
            return ok and self.player_tile() == dst
        # jump/fall macro: align to tile centre at rest, then replay
        if not self._settle() or not self._walk_to_x(src[0] + 0.5) or not self._settle():
            return False
        script = MACROS[kind]
        for a in script:
            if eng.done:
                return False
            self.act(a)
            if self.player_tile() == dst and eng._on_ground(eng.player):
                return True
        for _ in range(50):
            if eng.done:
                return False
            if eng._on_ground(eng.player):
                break
            self.act(Actions())
        return self.player_tile() == dst

    def goto(self, pos, tol=0.5, max_ticks=4000) -> bool:
        eng = self.eng
        goal_tile = (int(pos[0]), int(pos[1]))
        for attempt in range(4):
            if eng.done:
                return False
            if self._dist_to(pos) <= tol:
                return True
            if not self._settle():
                return False
            graph = self._graph()
            start = self.player_tile()
            if start not in graph:
                # straddling a ledge edge: shuffle onto a neighbouring
                # standing tile and re-anchor
                c, r = start
                cands = [t for t in graph if abs(t[0] - c) <= 1 and abs(t[1] - r) <= 2]
                if not cands:
                    return False
                t = min(cands, key=lambda t: abs(t[0] + 0.5 - eng.player.x))
                self._walk_to_x(t[0] + 0.5)
                self._settle()
                start = self.player_tile()
                if start not in graph:
                    return False
            path = self._graph_bfs(graph, start, goal_tile)
            if path is None:
                return False
            ok = True
            for (src, dst, kind) in path:
                if not self._do_edge(src, dst, kind):
                    ok = False
                    break
                if self._dist_to(pos) <= tol:
                    return True
            if ok:
                self._walk_to_x(pos[0], tol=0.12, max_ticks=40)
                if self._dist_to(pos) <= tol or self.player_tile() == goal_tile:
                    return True
        return self._dist_to(pos) <= tol

    @staticmethod
    def _graph_bfs(graph, start, goal):
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while prev[cur] is not None:
                    p, kind = prev[cur]
                    path.append((p, cur, kind))
                    cur = p
                return path[::-1]
            for nxt, (kind, cost) in graph.get(cur, {}).items():
                if nxt not in prev:
                    prev[nxt] = (cur, kind)
                    q.append(nxt)
        return None

    def open_door(self, door) -> bool:
        eng = self.eng
        graph = self._graph()
        # approach from either side; on stepped terrain the standing tile
        # next to the door can sit a row or two off the door's own row
        cands = []
        for side in (-1, 1):
            for dr in (0, 1, -1, 2):
                adj = (door.c + side, door.r + dr)
                if adj in graph:
                    cands.append((side, adj))
                    break
        cands.sort(key=lambda sa: self._dist_to((sa[1][0] + 0.5, sa[1][1] + 0.5)))
        for side, adj in cands:
            if not self.goto((adj[0] + 0.5, adj[1] + 0.5), tol=0.4):
                continue
            for _ in range(40):
                if door.open or eng.done:
                    break
                self.act(Actions(left=side > 0, right=side < 0))
            if door.open:
                self._graph_cache = None  # world changed; rebuild movement graph
                return True
        return door.open

    def press_plate(self, plate) -> bool:
        return False  # crates/plates are topdown-only features
