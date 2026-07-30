"""Human play in the terminal (curses, stdlib): WASD/arrows + space to jump."""

from __future__ import annotations

import curses
import time

from .engine import Engine, Actions, DT
from .spec import Scene

KEYS_HELD_TICKS = 4  # a keypress counts as held for this many ticks


def play(scene: Scene):
    result = {}
    curses.wrapper(_loop, scene, result)
    return result


def _loop(stdscr, scene: Scene, result: dict):
    curses.curs_set(0)
    stdscr.nodelay(True)
    eng = Engine(scene)
    held = {"L": 0, "R": 0, "U": 0, "D": 0, "J": 0}
    last = time.monotonic()
    while not eng.done:
        # drain input
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            if ch in (ord("q"), 27):
                result.update(success=False, reason="quit", ticks=eng.tick)
                return
            for k, codes in (("L", (curses.KEY_LEFT, ord("a"))),
                             ("R", (curses.KEY_RIGHT, ord("d"))),
                             ("U", (curses.KEY_UP, ord("w"))),
                             ("D", (curses.KEY_DOWN, ord("s"))),
                             ("J", (ord(" "), ord("k")))):
                if ch in codes:
                    held[k] = KEYS_HELD_TICKS
        act = Actions(held["L"] > 0, held["R"] > 0, held["U"] > 0,
                      held["D"] > 0, held["J"] > 0)
        for k in held:
            held[k] = max(0, held[k] - 1)
        eng.step(act)
        _draw(stdscr, eng)
        # pace to real time
        now = time.monotonic()
        sleep = DT - (now - last)
        if sleep > 0:
            time.sleep(sleep)
        last = time.monotonic()
    _draw(stdscr, eng, final=True)
    stdscr.nodelay(False)
    stdscr.getch()
    result.update(success=eng.success, reason="played", ticks=eng.tick)


def _draw(stdscr, eng: Engine, final=False):
    from .render import ascii_frame
    stdscr.erase()
    header = (f" {eng.scene.name} | hp {'♥' * eng.hp}{'♡' * max(0, 3 - eng.hp)} "
              f"| {eng.time:5.1f}s | keys: "
              + (",".join(f"{k}x{v}" for k, v in eng.inventory.items() if v) or "-")
              + " | move: WASD/arrows, jump: space, quit: q")
    try:
        stdscr.addstr(0, 0, header[: curses.COLS - 1])
        for i, line in enumerate(ascii_frame(eng).split("\n")):
            if i + 1 >= curses.LINES - 3:
                break
            stdscr.addstr(i + 1, 0, line[: curses.COLS - 1])
        row = min(curses.LINES - 2, eng.H + 2)
        objs = "  ".join(("✓" if ob.status == "passed" else "✗" if ob.status == "failed" else "○")
                         + ob.spec.describe() for ob in eng.objectives)
        stdscr.addstr(row, 0, objs[: curses.COLS - 1])
        if final:
            msg = " YOU WIN!  (any key) " if eng.success else " GAME OVER  (any key) "
            stdscr.addstr(row + 1, 0, msg[: curses.COLS - 1], curses.A_REVERSE)
    except curses.error:
        pass  # terminal too small; keep simulating
    stdscr.refresh()
