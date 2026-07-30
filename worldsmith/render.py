"""Renderers: ASCII (terminal + LLM observation), PNG (stdlib-only), and a
self-contained HTML replay viewer with a canvas animation of the whole run.

No third-party imports anywhere: PNG is raw zlib + struct, and the animated
GIF writer implements LZW compression from scratch.
"""

from __future__ import annotations

import json
import struct
import zlib

from .engine import Engine, Actions, replay
from .spec import Scene, COLORS

# ------------------------------------------------------------------ ASCII

ENTITY_CHARS = {"player": "P", "goal": "G", "coin": "o", "key": "k",
                "crate": "C", "plate": "_", "enemy": "E"}


def ascii_frame(eng: Engine, legend: bool = False) -> str:
    grid = [list(row) for row in eng.scene.tiles]

    def put(x, y, ch):
        c, r = int(x), int(y)
        if 0 <= c < eng.W and 0 <= r < eng.H:
            grid[r][c] = ch

    for pl in eng.plates:
        put(pl.x, pl.y, "*" if pl.latched else "_")
    for d in eng.doors:
        grid[d.r][d.c] = "/" if d.open else "D"
    for it in eng.items:
        if not it.taken:
            put(it.x, it.y, "k" if it.kind == "key" else "o")
    for crate in eng.crates:
        put(crate.x, crate.y, "C")
    for g in eng.goals:
        put(g.x, g.y, "G")
    for e in eng.enemies:
        put(e.body.x, e.body.y, "E")
    put(eng.player.x, eng.player.y, "P")
    out = "\n".join("".join(row) for row in grid)
    if legend:
        out += ("\n# wall  . floor  ~ lava  ^ spikes  P player  G goal  o coin"
                "\nk key  D closed-door  / open-door  C crate  _ plate  * latched-plate  E enemy")
    return out


# -------------------------------------------------------------------- PNG

PALETTE = {
    "#": (52, 57, 70), ".": (170, 178, 189), "~": (232, 93, 38), "^": (120, 124, 134),
    "bg": (28, 30, 38),
    "player": (36, 130, 228), "goal": (46, 184, 92), "coin": (240, 195, 55),
    "crate": (146, 100, 54), "plate": (108, 88, 130), "enemy": (196, 44, 74),
    "red": (214, 60, 60), "blue": (66, 108, 224), "green": (58, 168, 82),
    "yellow": (222, 186, 48), "purple": (150, 82, 190), "orange": (230, 130, 42),
}
SCALE = 26


class _Raster:
    """RGB pixel buffer -> PNG. Colors are (r, g, b) tuples."""

    BPP = 3

    def __init__(self, w, h, color):
        self.w, self.h = w, h
        self.px = bytearray(self._enc(color) * (w * h))

    def _enc(self, color):
        return bytes(color)

    def rect(self, x0, y0, x1, y1, color):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(x1)), min(self.h, int(y1))
        row = self._enc(color) * max(0, x1 - x0)
        for y in range(y0, y1):
            off = (y * self.w + x0) * self.BPP
            self.px[off:off + len(row)] = row

    def disc(self, cx, cy, rad, color):
        r2 = rad * rad
        val = self._enc(color)
        for y in range(int(cy - rad), int(cy + rad) + 1):
            if not (0 <= y < self.h):
                continue
            for x in range(int(cx - rad), int(cx + rad) + 1):
                if 0 <= x < self.w and (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    off = (y * self.w + x) * self.BPP
                    self.px[off:off + self.BPP] = val

    def to_png(self) -> bytes:
        raw = b"".join(b"\x00" + bytes(self.px[y * self.w * 3:(y + 1) * self.w * 3])
                       for y in range(self.h))

        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6))
                + chunk(b"IEND", b""))


class _IndexRaster(_Raster):
    """Palette-indexed buffer (1 byte/pixel) for GIF. Takes the same (r,g,b)
    colors as _Raster so the drawing code is written once."""

    BPP = 1

    def __init__(self, w, h, color, index_of):
        self.index_of = index_of
        super().__init__(w, h, color)

    def _enc(self, color):
        return bytes((self.index_of[tuple(color)],))


def _draw_scene(img, scene: Scene, eng: Engine, S: int):
    """Paint one frame of engine state. Shared by the PNG and GIF writers."""
    tile_colors = dict(PALETTE)
    if scene.mode == "platformer":     # dark sky, light solid ground
        tile_colors["."] = (34, 37, 47)
        tile_colors["#"] = (150, 158, 172)
    for r in range(eng.H):
        for c in range(eng.W):
            ch = scene.tiles[r][c]
            col = tile_colors.get(ch, tile_colors["."])
            img.rect(c * S + 1, r * S + 1, (c + 1) * S - 1, (r + 1) * S - 1, col)
            if ch == "^":
                img.rect(c * S + S // 4, r * S + S // 3, c * S + 3 * S // 4,
                         r * S + 2 * S // 3, SPIKE_DARK)
    for pl in eng.plates:
        col = PALETTE.get(pl.color, PALETTE["plate"])
        img.rect(pl.x * S - S * 0.32, pl.y * S - S * 0.32,
                 pl.x * S + S * 0.32, pl.y * S + S * 0.32, col)
        # hollow while unpressed, solid once a crate latches it
        if not pl.latched:
            img.rect(pl.x * S - S * 0.2, pl.y * S - S * 0.2,
                     pl.x * S + S * 0.2, pl.y * S + S * 0.2, tile_colors["."])
    for d in eng.doors:
        col = PALETTE.get(d.color, PALETTE["plate"])
        if d.open:
            img.rect(d.c * S + 1, d.r * S + 1, (d.c + 1) * S - 1, (d.r + 1) * S - 1,
                     tile_colors["."])
            img.rect(d.c * S + 1, d.r * S + 1, d.c * S + max(2, S // 5),
                     (d.r + 1) * S - 1, col)
        else:
            img.rect(d.c * S + 1, d.r * S + 1, (d.c + 1) * S - 1, (d.r + 1) * S - 1, col)
            img.rect(d.c * S + S // 3, d.r * S + S // 3, d.c * S + 2 * S // 3,
                     d.r * S + 2 * S // 3, _darken(col))
    for g in eng.goals:
        img.rect(g.x * S - S * 0.4, g.y * S - S * 0.4, g.x * S + S * 0.4,
                 g.y * S + S * 0.4, PALETTE["goal"])
    for it in eng.items:
        if it.taken:
            continue
        if it.kind == "coin":
            img.disc(it.x * S, it.y * S, S * 0.22, PALETTE["coin"])
        else:
            col = PALETTE.get(it.color, PALETTE["coin"])
            img.rect(it.x * S - S * 0.1, it.y * S - S * 0.3, it.x * S + S * 0.1,
                     it.y * S + S * 0.25, col)
            img.disc(it.x * S, it.y * S - S * 0.22, S * 0.16, col)
    for crate in eng.crates:
        img.rect(crate.x * S - S * 0.4, crate.y * S - S * 0.4,
                 crate.x * S + S * 0.4, crate.y * S + S * 0.4, PALETTE["crate"])
    for e in eng.enemies:
        img.disc(e.body.x * S, e.body.y * S, S * 0.36, PALETTE["enemy"])
    img.disc(eng.player.x * S, eng.player.y * S, S * 0.34, PALETTE["player"])


def _darken(col):
    return tuple(max(0, v - 60) for v in col)


SPIKE_DARK = (30, 30, 34)


def scene_png(scene: Scene, path: str, eng: Engine = None):
    """Snapshot of the scene (or a live engine state) as a PNG file."""
    eng = eng or Engine(scene)
    img = _Raster(eng.W * SCALE, eng.H * SCALE, PALETTE["bg"])
    _draw_scene(img, scene, eng, SCALE)
    with open(path, "wb") as f:
        f.write(img.to_png())


# -------------------------------------------------------------- animated GIF
# A from-scratch GIF89a writer: the drawing palette becomes the global color
# table, frames are LZW-compressed (implemented below), and each frame after
# the first is cropped to its changed bounding box with unchanged pixels left
# transparent. On flat game graphics with a small moving player that shrinks
# the file by well over an order of magnitude.

def _gif_palette():
    """Every RGB value the renderer can emit, as an ordered palette."""
    cols = list(PALETTE.values()) + [SPIKE_DARK, (34, 37, 47), (150, 158, 172)]
    cols += [_darken(PALETTE[c]) for c in COLORS] + [_darken(PALETTE["plate"])]
    seen, pal = set(), []
    for c in cols:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            pal.append(t)
    return pal


def _lzw_encode(data: bytes, min_code_size: int) -> bytes:
    """GIF-flavoured LZW: variable-width codes packed LSB-first."""
    clear, end = 1 << min_code_size, (1 << min_code_size) + 1
    code_size = min_code_size + 1
    table, next_code = {}, end + 1
    out, bitbuf, bitcnt = bytearray(), 0, 0

    def emit(code):
        nonlocal bitbuf, bitcnt
        bitbuf |= code << bitcnt
        bitcnt += code_size
        while bitcnt >= 8:
            out.append(bitbuf & 0xFF)
            bitbuf >>= 8
            bitcnt -= 8

    emit(clear)
    if data:
        prefix = data[0]
        for ch in data[1:]:
            key = (prefix, ch)
            if key in table:
                prefix = table[key]
                continue
            emit(prefix)
            table[key] = next_code
            next_code += 1
            if next_code > (1 << code_size):
                if code_size < 12:
                    code_size += 1
                else:                      # table full: restart the dictionary
                    emit(clear)
                    table.clear()
                    next_code = end + 1
                    code_size = min_code_size + 1
            prefix = ch
        emit(prefix)
    emit(end)
    if bitcnt:
        out.append(bitbuf & 0xFF)
    # LZW output ships as sub-blocks of at most 255 bytes, zero-terminated
    blocks = bytearray()
    for i in range(0, len(out), 255):
        piece = out[i:i + 255]
        blocks.append(len(piece))
        blocks += piece
    blocks.append(0)
    return bytes(blocks)


def _gif_bytes(frames, w, h, pal, delays, loop: bool = True) -> bytes:
    """Assemble indexed frames (bytearrays of palette indices) into a GIF.
    `delays` is one centisecond delay per frame."""
    transparent = len(pal)                       # one spare slot for deltas
    bits = max(2, (len(pal) + 1 - 1).bit_length())
    table = bytearray()
    for i in range(1 << bits):
        table += bytes(pal[i]) if i < len(pal) else b"\x00\x00\x00"

    out = bytearray(b"GIF89a")
    out += struct.pack("<HHBBB", w, h, 0xF0 | (bits - 1), 0, 0)
    out += table
    if loop:
        out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"

    prev = None
    for fr, delay_cs in zip(frames, delays):
        if prev is None:
            sub, x0, y0, sw, sh, tflag = fr, 0, 0, w, h, 0
        else:
            # changed-pixel bounding box
            xs0, ys0, xs1, ys1 = w, h, -1, -1
            for y in range(h):
                base = y * w
                if fr[base:base + w] == prev[base:base + w]:
                    continue
                ys0, ys1 = min(ys0, y), max(ys1, y)
                for x in range(w):
                    if fr[base + x] != prev[base + x]:
                        xs0 = min(xs0, x)
                        break
                for x in range(w - 1, -1, -1):
                    if fr[base + x] != prev[base + x]:
                        xs1 = max(xs1, x)
                        break
            if ys1 < 0:                          # identical frame: hold it
                xs0, ys0, xs1, ys1 = 0, 0, 0, 0
            x0, y0, sw, sh = xs0, ys0, xs1 - xs0 + 1, ys1 - ys0 + 1
            sub = bytearray(sw * sh)
            for y in range(sh):
                src = (y0 + y) * w + x0
                dst = y * sw
                for x in range(sw):
                    a, b = fr[src + x], prev[src + x]
                    sub[dst + x] = a if a != b else transparent
            tflag = 1
        # disposal method 1 = leave the frame in place, so the next frame's
        # transparent pixels reveal it (this is what makes the delta work)
        out += b"\x21\xF9\x04" + bytes((0x04 | tflag,)) + struct.pack("<H", delay_cs) \
            + bytes((transparent, 0))
        out += b"\x2C" + struct.pack("<HHHHB", x0, y0, sw, sh, 0)
        out += bytes((bits,)) + _lzw_encode(bytes(sub), bits)
        prev = fr
    out += b"\x3B"
    return bytes(out)


def replay_gif(scene: Scene, trace: list, path: str, scale: int = 14,
               stride: int = 3, max_frames: int = 260, hold_cs: int = 120):
    """Animate a recorded action trace as a self-contained GIF."""
    pal = _gif_palette()
    index_of = {c: i for i, c in enumerate(pal)}
    w, h = scene.width * scale, scene.height * scale
    # keep the file sane for long traces by widening the stride
    stride = max(stride, -(-len(trace) // max(1, max_frames)))

    def shot(eng):
        img = _IndexRaster(w, h, PALETTE["bg"], index_of)
        _draw_scene(img, scene, eng, scale)
        return img.px

    eng = Engine(scene)
    frames = [shot(eng)]
    for i, act in enumerate(trace):
        if eng.done:
            break
        eng.step(act)
        if (i + 1) % stride == 0:
            frames.append(shot(eng))
    frames.append(shot(eng))                      # final state
    step_cs = max(2, round(stride * 100 / 30))    # play back at wall-clock rate
    delays = [step_cs] * len(frames)
    delays[-1] = hold_cs                          # linger on the solved frame
    data = _gif_bytes(frames, w, h, pal, delays)
    with open(path, "wb") as f:
        f.write(data)
    return len(frames), len(data)


# ------------------------------------------------------------ HTML replay

def record_frames(scene: Scene, trace: list, max_frames: int = 1500):
    """Re-simulate a trace, capturing dynamic state for the viewer. Long
    traces are subsampled (every stride-th tick plus the final state) so the
    embedded JSON stays small; the viewer scales times by the stride."""
    stride = max(1, -(-len(trace) // max_frames))
    eng = Engine(scene)
    frames = [_frame(eng)]
    for i, act in enumerate(trace):
        if eng.done:
            break
        eng.step(act)
        if (i + 1) % stride == 0:
            frames.append(_frame(eng))
    if len(trace) % stride:
        frames.append(_frame(eng))
    return frames, eng, stride


def _frame(eng: Engine):
    return {
        "p": [round(eng.player.x, 3), round(eng.player.y, 3)],
        "hp": eng.hp,
        "items": [1 if it.taken else 0 for it in eng.items],
        "doors": [1 if d.open else 0 for d in eng.doors],
        "plates": [1 if pl.latched else 0 for pl in eng.plates],
        "crates": [[round(c.x, 3), round(c.y, 3)] for c in eng.crates],
        "enemies": [[round(e.body.x, 3), round(e.body.y, 3)] for e in eng.enemies],
        # o = open (pending), p = passed, f = failed
        "obj": [{"pending": "o", "passed": "p", "failed": "f"}[ob.status]
                for ob in eng.objectives],
    }


def replay_html(scene: Scene, trace: list, path: str, title: str = ""):
    frames, eng, stride = record_frames(scene, trace)
    data = {
        "title": title or scene.name,
        "stride": stride,
        "mode": scene.mode,
        "W": scene.width, "H": scene.height,
        "tiles": scene.tiles,
        "statics": {
            "goals": [[g.x, g.y] for g in eng.goals],
            "items": [[it.x, it.y, it.kind, it.color] for it in eng.items],
            "doors": [[d.c, d.r, d.color] for d in eng.doors],
            "plates": [[pl.x, pl.y, pl.color] for pl in eng.plates],
        },
        "objectives": [ob.spec.describe() for ob in eng.objectives],
        "events": [[t, k, d] for (t, k, d) in eng.events
                   if k in ("key_collected", "door_opened", "coin_collected",
                            "plate_pressed", "damage", "death", "goal_touched",
                            "objective_passed", "episode_end")],
        "success": eng.success,
        "frames": frames,
    }
    # <-escape so hostile strings (e.g. a scene named "</script>...")
    # can never terminate the inline data script and inject markup
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = _HTML_TEMPLATE.replace("__DATA__", payload)
    with open(path, "w") as f:
        f.write(html)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WorldSmith replay</title>
<style>
 body{background:#16171d;color:#cfd3dc;font:14px/1.5 -apple-system,Segoe UI,sans-serif;
      display:flex;flex-direction:column;align-items:center;margin:0;padding:18px}
 h1{font-size:17px;margin:4px 0 10px;color:#e8eaf0}
 #wrap{display:flex;gap:16px;flex-wrap:wrap;justify-content:center}
 canvas{background:#1c1e26;border-radius:8px;max-width:96vw}
 #side{width:280px;font-size:13px}
 .card{background:#1e2029;border-radius:8px;padding:10px 12px;margin-bottom:10px}
 .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:#8b92a5;margin:0 0 6px}
 #controls{display:flex;gap:8px;align-items:center;margin-top:10px;width:100%;max-width:640px}
 button{background:#2d3040;border:0;color:#e8eaf0;border-radius:6px;padding:6px 14px;
        font-size:14px;cursor:pointer} button:hover{background:#3a3e52}
 input[type=range]{flex:1}
 .ev{color:#9aa1b4;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .ok{color:#4cc36d}.bad{color:#e05555}
 #status{font-weight:600}
</style></head><body>
<h1 id="title"></h1>
<div id="wrap">
 <div><canvas id="cv"></canvas>
  <div id="controls">
   <button id="play">&#9654;</button>
   <input type="range" id="seek" min="0" value="0">
   <span id="clock" style="min-width:90px;text-align:right"></span>
  </div>
 </div>
 <div id="side">
  <div class="card"><h2>Objectives</h2><div id="obj"></div></div>
  <div class="card"><h2>State</h2><div id="state"></div></div>
  <div class="card"><h2>Events</h2><div id="events" style="max-height:180px;overflow:auto"></div></div>
 </div>
</div>
<script>
const D=__DATA__;
const COLORS={red:"#d63c3c",blue:"#426ce0",green:"#3aa852",yellow:"#deba30",
  purple:"#9652be",orange:"#e6822a"};
const S=Math.max(12,Math.min(30,Math.floor(920/D.W)));
const cv=document.getElementById("cv");cv.width=D.W*S;cv.height=D.H*S;
const g=cv.getContext("2d");
document.getElementById("title").textContent=D.title+"  ·  "+D.mode+"  ·  "+
  (D.success?"SOLVED":"not solved")+" in "+((D.frames.length-1)*D.stride/30).toFixed(1)+"s";
let f=0,playing=false,speed=1;
const seek=document.getElementById("seek");seek.max=D.frames.length-1;
function drawTile(c,r,ch){
 if(ch=="#")g.fillStyle=D.mode=="platformer"?"#969eac":"#343946";
 else if(ch=="~")g.fillStyle="#e85d26";
 else if(ch=="^")g.fillStyle="#787c86";
 else g.fillStyle=D.mode=="platformer"?"#20222c":"#aab2bd";
 g.fillRect(c*S,r*S,S,S);
 if(ch=="^"){g.fillStyle="#23252e";
  g.beginPath();g.moveTo(c*S+S*.2,r*S+S*.85);g.lineTo(c*S+S*.5,r*S+S*.2);
  g.lineTo(c*S+S*.8,r*S+S*.85);g.fill();}
 if(ch=="."&&D.mode!="platformer"){g.strokeStyle="rgba(0,0,0,.05)";g.strokeRect(c*S,r*S,S,S);}
}
function draw(){
 const fr=D.frames[f];
 for(let r=0;r<D.H;r++)for(let c=0;c<D.W;c++)drawTile(c,r,D.tiles[r][c]);
 D.statics.plates.forEach((p,i)=>{g.fillStyle=COLORS[p[2]]||"#6c5882";
  g.fillRect((p[0]-.34)*S,(p[1]-.34)*S,S*.68,S*.68);
  g.fillStyle=fr.plates[i]?COLORS[p[2]]:"#aab2bd";
  g.fillRect((p[0]-.2)*S,(p[1]-.2)*S,S*.4,S*.4);});
 D.statics.doors.forEach((d,i)=>{const col=COLORS[d[2]]||"#888";
  if(fr.doors[i]){g.fillStyle=col;g.fillRect(d[0]*S,d[1]*S,S*.15,S);}
  else{g.fillStyle=col;g.fillRect(d[0]*S,d[1]*S,S,S);
   g.fillStyle="rgba(0,0,0,.35)";g.fillRect(d[0]*S+S*.33,d[1]*S+S*.33,S*.34,S*.34);}});
 D.statics.goals.forEach(gl=>{g.fillStyle="#2eb85c";
  g.fillRect((gl[0]-.4)*S,(gl[1]-.4)*S,S*.8,S*.8);
  g.fillStyle="#9be3b2";g.fillRect((gl[0]-.16)*S,(gl[1]-.16)*S,S*.32,S*.32);});
 D.statics.items.forEach((it,i)=>{if(fr.items[i])return;
  if(it[2]=="coin"){g.fillStyle="#f0c337";g.beginPath();
   g.arc(it[0]*S,it[1]*S,S*.22,0,7);g.fill();}
  else{g.fillStyle=COLORS[it[3]]||"#f0c337";
   g.beginPath();g.arc(it[0]*S,(it[1]-.18)*S,S*.15,0,7);g.fill();
   g.fillRect((it[0]-.07)*S,(it[1]-.15)*S,S*.14,S*.4);}});
 fr.crates.forEach(c=>{g.fillStyle="#926436";g.fillRect((c[0]-.4)*S,(c[1]-.4)*S,S*.8,S*.8);
  g.strokeStyle="#6b4a28";g.strokeRect((c[0]-.4)*S,(c[1]-.4)*S,S*.8,S*.8);});
 fr.enemies.forEach(e=>{g.fillStyle="#c42c4a";g.beginPath();
  g.arc(e[0]*S,e[1]*S,S*.36,0,7);g.fill();
  g.fillStyle="#fff";g.fillRect((e[0]-.18)*S,(e[1]-.12)*S,S*.1,S*.1);
  g.fillRect((e[0]+.08)*S,(e[1]-.12)*S,S*.1,S*.1);});
 g.fillStyle="#2482e4";g.beginPath();g.arc(fr.p[0]*S,fr.p[1]*S,S*.34,0,7);g.fill();
 g.fillStyle="#cfe3fa";g.beginPath();g.arc(fr.p[0]*S,(fr.p[1]-.08)*S,S*.12,0,7);g.fill();
 document.getElementById("clock").textContent=(f*D.stride/30).toFixed(2)+"s / "+
   ((D.frames.length-1)*D.stride/30).toFixed(1)+"s";
 seek.value=f;
 let hp="&#10084;".repeat(fr.hp)+"&#9825;".repeat(Math.max(0,3-fr.hp));
 document.getElementById("state").innerHTML=
  "hp "+hp+"<br>tick "+(f*D.stride)+"<br>player ("+fr.p[0].toFixed(1)+", "+fr.p[1].toFixed(1)+")";
 document.getElementById("obj").innerHTML=D.objectives.map((o,i)=>{
  const st=fr.obj[i];
  const mark=st=="f"?"&#10007;":st=="p"?"&#10003;":"&#9711;";
  const cls=st=="p"?"ok":st=="f"?"bad":"";
  return "<div class='"+cls+"'>"+mark+" "+o+"</div>";}).join("");
 const evs=D.events.filter(e=>e[0]<=f*D.stride).slice(-14);
 document.getElementById("events").innerHTML=evs.map(e=>
  "<div class='ev'>t"+e[0]+" "+e[1]+" "+JSON.stringify(e[2]).replaceAll('"','')+"</div>").join("");
}
function tickLoop(){if(playing){f=Math.min(D.frames.length-1,f+speed);
 if(f>=D.frames.length-1)playing=false,updateBtn();draw();}requestAnimationFrame(tickLoop);}
function updateBtn(){document.getElementById("play").innerHTML=playing?"&#10074;&#10074;":"&#9654;";}
document.getElementById("play").onclick=()=>{if(f>=D.frames.length-1)f=0;
 playing=!playing;updateBtn();};
seek.oninput=()=>{f=+seek.value;playing=false;updateBtn();draw();};
draw();tickLoop();
</script></body></html>
"""
