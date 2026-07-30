"""Rules brain: compile a free-text command into a GenPlan, offline.

This is the deterministic fallback generation backend. It understands the
domain vocabulary (modes, layouts, features, counts, sizes, difficulty) well
enough to cover most reasonable commands without any LLM. The LLM brain
(brain_llm.py) produces the same GenPlan structure with real language
understanding; both feed the identical procgen + verification pipeline.
"""

from __future__ import annotations

import re

from .procgen import GenPlan

_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2,
    "few": 3, "several": 4, "some": 3, "many": 6, "lots": 8,
}

# the captured token must BE a number word, so an unrelated preceding word
# can never consume the match ("arena with 5 monsters" must see the 5)
_NUM_PAT = r"\b(\d+|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")\b"
# intermediate adjectives may sit between number and noun ("3 locked doors",
# "a 3-door dungeon"), but never structural words; otherwise "level 3 with a
# door" would read as 3 doors
_FILLER = r"(?:(?!with\b|and\b|no\b|without\b|the\b|then\b|plus\b)\w+[-\s]+){0,2}?"
_NEG_PAT = r"\b(?:no|without|zero|not?\s+any|free\s+of)\s+" + _FILLER


def _count_before(text: str, nouns, default: int = 1) -> int:
    """Find 'three keys', '2 doors', 'a 3-door dungeon' style counts.
    The noun tail is a lookahead so candidate numbers can overlap; otherwise
    'a 3-door dungeon' consumes 'a ... door' and never sees the 3."""
    best = None
    for noun in nouns:
        pat = _NUM_PAT + r"(?=[-\s]+(?:of\s+)?" + _FILLER + noun + r")"
        for m in re.finditer(pat, text):
            w = m.group(1).lower()
            n = int(w) if w.isdigit() else _NUM_WORDS[w]
            best = max(best or 0, n)
    if best is None:
        best = default
    return best


def _mentions(text: str, *words) -> bool:
    return any(re.search(r"\b" + w, text) for w in words)


def _negated(text: str, *nouns) -> bool:
    """True when the command excludes a feature ('no doors', 'without lava')."""
    return any(re.search(_NEG_PAT + n, text) for n in nouns)


def compile_text(text: str) -> GenPlan:
    t = text.lower()
    plan = GenPlan()
    plan.name = _slug(text)
    recognized = [False]  # did ANY vocabulary match? (else warn: all defaults)

    def saw(*words):
        hit = _mentions(t, *words)
        if hit:
            recognized[0] = True
        return hit

    # --- mode / layout
    if saw("platform", "side.?scroll", "jump", "gravity", "mario"):
        plan.mode = "platformer"
        plan.layout = "terrain"
    elif saw("maze", "labyrinth"):
        plan.layout = "maze"
    elif saw("arena", "open room", "field", "courtyard"):
        plan.layout = "arena"
    elif saw("dungeon", "room", "cave", "crypt", "castle", "level"):
        plan.layout = "dungeon"
    else:
        plan.layout = "dungeon"

    # --- size
    if saw("tiny", "minuscule"):
        plan.size = (16, 10) if plan.mode == "topdown" else (24, 12)
    elif saw("small", "little", "compact", "cozy"):
        plan.size = (20, 12) if plan.mode == "topdown" else (32, 14)
    elif saw("large", "big", "huge", "giant", "massive", "enormous", "vast",
             "grand", "sprawling", "long"):
        plan.size = (40, 22) if plan.mode == "topdown" else (56, 16)
    else:
        plan.size = (26, 15) if plan.mode == "topdown" else (40, 14)
    m = re.search(r"(\d{1,3})\s*(?:x|by)\s*(\d{1,3})", t)
    if m:
        recognized[0] = True
        plan.size = (int(m.group(1)), int(m.group(2)))

    def capped(noun_label, want, cap):
        if want > cap:
            plan.warnings.append(f"requested {want} {noun_label}, capped at {cap}")
        return min(cap, want)

    # --- features
    if saw("door", "lock", "key", "gate"):
        plan.doors = capped("locked doors/keys", _count_before(t, ("door", "lock", "key", "gate"), 1), 3)
    if saw("crate", "plate", "button", "box.?puzzle", "push", "sokoban", "switch"):
        plan.plate = True
    if saw("coin", "treasure", "gem", "collect", "loot", "gold"):
        plan.coins = capped("coins/treasures", _count_before(t, ("coin", "treasure", "gem", "gold"), 3), 12)
        plan.collect_coins = _mentions(t, "collect", "all", "every", "gather")
        if not plan.collect_coins and plan.coins:
            plan.collect_coins = True  # coins exist to be collected
    if saw("enem", "guard", "monster", "patrol", "creature", "skeleton", "zombie"):
        plan.enemies = capped("enemies", _count_before(
            t, ("enem", "guard", "monster", "patrol", "creature", "skeleton", "zombie"), 2), 6)
    if saw("lava", "fire", "magma"):
        plan.lava = "moat" if _mentions(t, "moat", "ring", "surround") else "blobs"
    if saw("spike", "pit", "trap") and plan.mode == "platformer":
        plan.spike_pits = min(5, _count_before(t, ("spike", "pit", "trap"), 2))
    elif saw("spike", "trap"):  # topdown spikes read as lava-style blobs
        if plan.lava == "none":
            plan.lava = "blobs"
    if plan.mode == "platformer" and plan.spike_pits == 0 and _mentions(t, "gap", "chasm"):
        plan.spike_pits = 2

    # --- difficulty nudges
    if saw("hard", "difficult", "challenging", "brutal", "danger"):
        plan.enemies = max(plan.enemies, 2 if plan.mode == "topdown" else 0)
        if plan.mode == "platformer":
            plan.spike_pits = max(plan.spike_pits, 3)
        elif plan.lava == "none":
            plan.lava = "blobs"
    if saw("easy", "simple", "gentle", "peaceful", "calm"):
        plan.enemies = 0
        plan.lava = "none"
        plan.spike_pits = min(plan.spike_pits, 1)

    m = re.search(r"(\d+)\s*(?:second|sec)s?\b", t)
    if m:
        plan.time_limit = float(m.group(1))
    m = re.search(r"(\d+)\s*(?:minute|min)s?\b", t)
    if m:
        plan.time_limit = float(m.group(1)) * 60

    # --- explicit exclusions win over mentions AND difficulty nudges:
    # "a hard dungeon with no monsters" must ship zero monsters
    if any(_negated(t, n) for n in ("door", "lock", "key", "gate", "enem", "guard",
                                    "monster", "lava", "spike", "coin", "treasure",
                                    "crate", "plate", "hazard", "danger")):
        recognized[0] = True
    if _negated(t, "door", "lock", "key", "gate"):
        plan.doors = 0
    if _negated(t, "enem", "guard", "monster", "patrol", "creature", "skeleton", "zombie"):
        plan.enemies = 0
    if _negated(t, "lava", "fire", "magma", "hazard", "danger"):
        plan.lava = "none"
    if _negated(t, "spike", "pit", "trap", "hazard", "danger"):
        plan.spike_pits = 0
    if _negated(t, "coin", "treasure", "gem", "gold", "loot"):
        plan.coins = 0
        plan.collect_coins = False
    if _negated(t, "crate", "plate", "button", "puzzle", "switch"):
        plan.plate = False

    if not recognized[0]:
        plan.warnings.append(
            "no recognized vocabulary in the command; generating a default dungeon")

    return plan.normalize()


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:5]
    return "-".join(words) or "environment"
