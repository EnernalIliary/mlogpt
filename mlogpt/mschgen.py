"""
mschgen.py -- turn mlog source into a ready-to-paste Mindustry blueprint (.msch).

Format verified against real files, e.g. 开关.msch:
    'msch' | 0x01 | zlib(
        int16 w, int16 h,
        uint8 tagCount,  tagCount * (UTF key, UTF value),
        uint8 dictLen,   dictLen  * UTF blockName,
        int32 total,
        total * { int8 dictIdx, int32 packed(x,y), <TypeIO config>, int8 rotation }
    )

A logic block's TypeIO config is type 14 (byte[]) whose payload is a SECOND
zlib stream:  uint8 version | int32 len | bytes | int32 links | links*(UTF, int16, int16)

Usage
-----
  # processor code from a file, link named dome1
  python mlogpt/mschgen.py logic.mlog -o out.msch --name "my-bp" \\
        --block world-processor --at 0,0 --link dome1@4,4

  # short code straight on the command line
  python mlogpt/mschgen.py --code "ubind @flare" -o out.msch

  # verify an existing blueprint round-trips
  python mlogpt/mschgen.py --selftest
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import zlib

try:                          # 作为包导入：python mlogpt/gen.py
    from . import msch         # 复用已验证的读写实现
except ImportError:            # 直接运行本文件时
    import msch

MAGIC = b"msch"
VERSION = 1

# Block sizes derived from the game source (see mlogpt/block_sizes.json). Sizes the
# preview frame -- the game recomputes it from the tiles on load.
_BLOCK_SIZES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "block_sizes.json")
try:
    with open(_BLOCK_SIZES_PATH, encoding="utf-8") as _fh:
        BLOCK_SIZES: dict[str, int] = json.load(_fh)
except Exception as _e:                       # missing file -> assume size 1
    print(f"mschgen: block_sizes.json unavailable ({_e}); assuming size 1",
          file=sys.stderr)
    BLOCK_SIZES = {}


def block_size(block: str) -> int:
    return int(BLOCK_SIZES.get(block, 1))

# Blocks we know how to emit a config for.
LOGIC_BLOCKS = {
    "micro-processor", "logic-processor", "hyper-processor",
    "world-processor",
}
MEMORY_BLOCKS = {"memory-cell", "memory-bank"}


class Link:
    """A named link from a processor. dx/dy are RELATIVE to the processor tile."""

    __slots__ = ("name", "dx", "dy")

    def __init__(self, name: str, dx: int, dy: int):
        self.name, self.dx, self.dy = name, int(dx), int(dy)


class Tile:
    __slots__ = ("block", "x", "y", "rotation", "config")

    def __init__(self, block: str, x: int, y: int, rotation: int = 0, config=None):
        self.block, self.x, self.y = block, int(x), int(y)
        self.rotation, self.config = int(rotation), config


# --------------------------------------------------------------------------- #
# TypeIO writer
# --------------------------------------------------------------------------- #

def write_object(out: bytearray, obj) -> None:
    """TypeIO.writeObject, matching mindustry.io.TypeIO:42."""
    if obj is None:
        out.append(0)                                   # null
    elif isinstance(obj, bool):
        out.append(10)                                  # bool
        out.append(1 if obj else 0)
    elif isinstance(obj, int):
        out.append(1)                                   # int
        out += struct.pack(">i", obj)
    elif isinstance(obj, float):
        out.append(3)                                   # float
        out += struct.pack(">f", obj)
    elif isinstance(obj, str):
        out.append(4)                                   # String
        out.append(1)
        sb = obj.encode("utf-8")
        out += struct.pack(">H", len(sb)) + sb
    elif isinstance(obj, (bytes, bytearray)):
        out.append(14)                                  # byte[]
        out += struct.pack(">i", len(obj))
        out += bytes(obj)
    elif isinstance(obj, dict) and "__content__" in obj:
        out.append(5)                                   # Content
        out.append(msch.CONTENT_TYPES.index(obj["__content__"]) & 0xFF)
        out += struct.pack(">h", obj["id"])
    elif isinstance(obj, dict) and "__point__" in obj:
        out.append(7)                                   # Point2
        out += struct.pack(">ii", *obj["__point__"])
    elif isinstance(obj, dict) and "__vec2__" in obj:
        out.append(19)                                  # Vec2 (floats)
        out += struct.pack(">ff", *obj["__vec2__"])
    elif isinstance(obj, dict) and "__team__" in obj:
        out.append(20)                                  # Team
        out.append(int(obj["__team__"]) & 0xFF)
    elif isinstance(obj, dict) and "__laccess__" in obj:
        out.append(13)                                  # LAccess
        out += struct.pack(">h", obj["__laccess__"])
    elif isinstance(obj, dict) and "__building__" in obj:
        out.append(12)                                  # Building
        out += struct.pack(">i", obj["__building__"])
    elif isinstance(obj, (list, tuple)) and all(isinstance(v, bool) for v in obj):
        out.append(16)                                  # boolean[]
        out += struct.pack(">i", len(obj))
        for v in obj:
            out.append(1 if v else 0)
    elif isinstance(obj, (list, tuple)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
        out.append(6)                                   # int[] / IntSeq
        out += struct.pack(">h", len(obj))
        for v in obj:
            out += struct.pack(">i", int(v))
    else:
        raise TypeError(f"cannot serialize config of type {type(obj).__name__}")


def write_utf(out: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    if len(b) > 0xFFFF:
        raise ValueError("string too long for UTF")
    out += struct.pack(">H", len(b)) + b


# --------------------------------------------------------------------------- #
# logic payload
# --------------------------------------------------------------------------- #

def build_logic_config(code: str, links: list[Link], level: int = 9) -> bytes:
    """Build the byte[] that goes in a logic block's TypeIO config field."""
    inner = bytearray()
    inner.append(1)                                     # logicVersion = 1
    cb = code.encode("utf-8")
    inner += struct.pack(">i", len(cb)) + cb
    inner += struct.pack(">i", len(links))
    for lk in links:
        write_utf(inner, lk.name)
        inner += struct.pack(">hh", lk.dx, lk.dy)
    return zlib.compress(bytes(inner), level)


# --------------------------------------------------------------------------- #
# blueprint assembly
# --------------------------------------------------------------------------- #

class Blueprint:
    """Compose a .msch from tiles. Add links per-logic-block with add_logic()."""

    def __init__(self, name: str = "blueprint", description: str = "",
                 labels: list[str] | None = None, deflate_level: int = 9):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.tiles: list[Tile] = []
        # link positions, kept only to size the preview box
        self.link_points: list[tuple[int, int]] = []
        # content ids are NOT serialized in a schematic (the game resolves a
        # block by its NAME on load), so this is informational only.
        self.content_ids: dict[str, int] = {}
        # zlib level for this blueprint: the per-tile logic configs and the body.
        # 9 is what the game emits and what generated blueprints must keep; the
        # self-test lowers it, because a throw-away round-trip only needs the
        # parsed content back (level 9 costs 0.200 s/MB, level 1 costs 0.008).
        self.deflate_level = deflate_level

    def add(self, block: str, x: int, y: int, rotation: int = 0, config=None,
            content_id: int | None = None) -> Tile:
        t = Tile(block, x, y, rotation, config)
        self.tiles.append(t)
        if content_id is not None:
            self.content_ids[block] = content_id
        return t

    def add_block(self, block: str, x: int, y: int, rotation: int = 0,
                  config=None, content_id: int | None = None) -> Tile:
        """Place a plain (non-logic) block, e.g. the overdrive dome a link targets."""
        if block in LOGIC_BLOCKS:
            raise ValueError(f"{block!r} is a logic block; use add_logic()")
        return self.add(block, x, y, rotation, config, content_id)

    def add_logic(self, code: str, x: int = 0, y: int = 0,
                  block: str = "micro-processor", rotation: int = 0,
                  links: list[Link] | None = None) -> Tile:
        """Place a processor. `links` coordinates must be relative to (x, y).

        NOTE: the link *target* building is NOT placed. A schematic only carries
        a link's name + relative position; the target must exist on the map for
        `dome1` to resolve to anything. Place the real block at that spot after
        pasting, or pass its content id via `with_block`.
        """
        if block not in LOGIC_BLOCKS:
            raise ValueError(f"{block!r} is not a logic block; "
                             f"expected one of {sorted(LOGIC_BLOCKS)}")
        # Normalise line endings only. Do NOT force a trailing newline:
        # real blueprints are inconsistent about it (some end with \n, some not)
        # and the game's parser does not care.
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        cfg = build_logic_config(code, links or [], self.deflate_level)
        for lk in (links or []):
            self.link_points.append((x + lk.dx, y + lk.dy))
        return self.add(block, x, y, rotation, cfg)

    def occupied(self) -> set[tuple[int, int]]:
        """Every tile covered by a placed block (centre-based footprints)."""
        cells: set[tuple[int, int]] = set()
        for t in self.tiles:
            sz = block_size(t.block)
            half = (sz - 1) // 2
            x0, y0 = t.x - half, t.y - half
            for dx in range(sz):
                for dy in range(sz):
                    cells.add((x0 + dx, y0 + dy))
        return cells

    def link_report(self) -> list[str]:
        """Report links whose target tile is not covered by any placed block.

        This is expected when you only generate a processor: a schematic carries
        a link's name + relative position, but the *target building must exist on
        the map* for the name (`dome1`) to resolve. Either place the block after
        pasting, or emit it with `--with-block`.
        """
        cells = self.occupied()
        out = []
        for p in self.link_points:
            if p not in cells:
                out.append(
                    f"link target at {p} is not covered by any placed block -- "
                    f"place the real building there after pasting, otherwise the "
                    f"link name resolves to nothing")
        return out

    def bounds(self) -> tuple[int, int]:
        """Bounding box of placed tiles, plus link positions.

        IMPORTANT: a Stile stores the block's CENTER tile, not its corner.
        Verified two ways:
          * source: Tile.setBlock() spreads a multiblock as
            `worldx = dx + -((size-1)/2) + x`  ->  (x,y) is the middle tile
          * empirically: across 624 real blueprints, treating the stored
            position as the corner produces block overlaps in 11 blueprints;
            treating it as the center produces 0.
        So the footprint of a size-N block centred at (x,y) spans
        [x - (N-1)//2, x - (N-1)//2 + N - 1] and likewise for y.
        """
        xs: list[int] = []
        ys: list[int] = []
        for t in self.tiles:
            sz = block_size(t.block)
            half = (sz - 1) // 2
            xs += [t.x - half, t.x - half + sz - 1]
            ys += [t.y - half, t.y - half + sz - 1]
        xs += [p[0] for p in self.link_points]
        ys += [p[1] for p in self.link_points]
        if not xs:
            return 1, 1
        return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1

    def _origin(self) -> tuple[int, int, int, int]:
        """Bounding box (minx, miny, maxx, maxy) in the caller's coordinates.

        Schematic tiles are written normalised (min at 0,0), so `build()` needs
        the origin to subtract; otherwise `--at 20,20` leaves the build offset
        inside a bigger box and the pasted copy lands 20 tiles away.
        """
        xs: list[int] = []
        ys: list[int] = []
        for t in self.tiles:
            sz = block_size(t.block)
            half = (sz - 1) // 2
            xs += [t.x - half, t.x - half + sz - 1]
            ys += [t.y - half, t.y - half + sz - 1]
        xs += [p[0] for p in self.link_points]
        ys += [p[1] for p in self.link_points]
        if not xs:
            return 0, 0, 0, 0
        return min(xs), min(ys), max(xs), max(ys)

    def build(self, level: int | None = None) -> bytes:
        """Serialize to .msch bytes (deflate level: self.deflate_level unless given)."""
        if not self.tiles:
            raise ValueError("blueprint has no tiles")

        # tags -- the game always writes these four
        tags = {
            "name": self.name,
            "contentMap": "{}",
            "description": self.description,
            "labels": _json_labels(self.labels),
        }

        # block dictionary: order of first appearance, like OrderedSet in Schematics.write
        dict_names: list[str] = []
        for t in self.tiles:
            if t.block not in dict_names:
                dict_names.append(t.block)

        w, h = self.bounds()
        ox, oy, _, _ = self._origin()
        body = bytearray()
        body += struct.pack(">hh", w, h)

        body.append(len(tags))
        for k, v in tags.items():
            write_utf(body, k)
            write_utf(body, v)

        body.append(len(dict_names))
        for n in dict_names:
            write_utf(body, n)

        body += struct.pack(">i", len(self.tiles))
        for t in self.tiles:
            body.append(dict_names.index(t.block) & 0xFF)
            # Schematic tile coordinates are normalised to the bounding box:
            # every real blueprint starts at (0,0). Writing raw center coords
            # (e.g. --at 20,20) leaves the content offset inside a larger box,
            # which shifts the pasted build and breaks the preview.
            body += struct.pack(">i", _pack(t.x - ox, t.y - oy))
            write_object(body, t.config)
            body.append(t.rotation & 0xFF)

        lvl = self.deflate_level if level is None else level
        return MAGIC + bytes([VERSION]) + zlib.compress(bytes(body), lvl)

    def write(self, path: str) -> str:
        data = self.build()
        with open(path, "wb") as fh:
            fh.write(data)
        return path


def _pack(x: int, y: int) -> int:
    return ((x & 0xFFFF) << 16) | (y & 0xFFFF)


def _json_labels(labels: list[str]) -> str:
    import json
    return json.dumps(labels, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# link parsing
# --------------------------------------------------------------------------- #

def parse_link(spec: str) -> Link:
    """`name@dx,dy` -> Link."""
    if "@" not in spec:
        raise ValueError(f"link {spec!r} must look like name@dx,dy")
    name, _, pos = spec.rpartition("@")
    try:
        dx, dy = (int(v) for v in pos.split(","))
    except ValueError:
        raise ValueError(f"bad link position in {spec!r}; expected name@dx,dy")
    return Link(name, dx, dy)


# --------------------------------------------------------------------------- #

def default_schematics_dir() -> str | None:
    """Locate Mindustry's schematics folder without hardcoding a user path.

    Order: $MINDUSTRY_SCHEMATICS  ->  %APPDATA%/Mindustry/schematics
           ->  ~/.local/share/Mindustry/schematics  ->  macOS location
    """
    env = os.environ.get("MINDUSTRY_SCHEMATICS")
    if env and os.path.isdir(env):
        return env
    cands = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        cands.append(os.path.join(appdata, "Mindustry", "schematics"))
    home = os.path.expanduser("~")
    cands += [
        os.path.join(home, ".local", "share", "Mindustry", "schematics"),
        os.path.join(home, "Library", "Application Support", "Mindustry", "schematics"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def _selftest_file(f: str) -> tuple[str, str, list[str]]:
    """One blueprint -> (logic_state, build_state, messages); "skip" = unusable.

    "skip" reproduces what the serial loop's two continue statements used to do
    (a file that will not parse, or that carries no logic tiles).  Module level,
    so ProcessPoolExecutor can pickle it.
    """
    try:
        sch = msch.parse_file(f)
    except Exception:
        return ("skip", "skip", [])
    if not sch.logic_tiles():
        return ("skip", "skip", [])

    msgs: list[str] = []
    logic_state = "ok"

    # ---- logic fidelity: rebuild a processor-only blueprint and compare ----
    try:
        lbp = Blueprint(sch.name or "t", sch.description, list(sch.labels),
                  deflate_level=1)
        for t in sch.logic_tiles():
            assert t.logic is not None
            lbp.add_logic(t.logic.code, t.x, t.y, t.block, t.rotation,
                          [Link(l.name, l.x - t.x, l.y - t.y)
                           for l in t.logic.links])
        # verification only: the rebuilt content is what gets compared, so the
        # deflate level of this throw-away blueprint is irrelevant
        back = msch.parse_bytes(lbp.build())

        # compare *relative layout*: a rebuilt blueprint is normalized to
        # (0,0), so pure translation must not count as a mismatch (the old
        # absolute compare scored 176/430 for exactly this reason).
        def _norm(items):
            if not items:
                return items
            ox = min(i[1] for i in items)
            oy = min(i[2] for i in items)
            return [(blk, x - ox, y - oy, rot, code,
                     [(n, lx - ox, ly - oy) for (n, lx, ly) in links])
                    for (blk, x, y, rot, code, links) in items]

        a = _norm([(t.block, t.x, t.y, t.rotation, t.logic.code,
                    [(l.name, l.x, l.y) for l in t.logic.links])
                   for t in sch.logic_tiles()])
        b = _norm([(t.block, t.x, t.y, t.rotation, t.logic.code,
                    [(l.name, l.x, l.y) for l in t.logic.links])
                   for t in back.logic_tiles()])
        if a != b:
            logic_state = "bad"
            msgs.append(f"logic mismatch: {os.path.basename(f)}")
    except Exception as e:
        logic_state = "bad"
        msgs.append(f"{os.path.basename(f)}: {type(e).__name__}: {e}")

    # ---- whole-blueprint build (best effort) ----
    try:
        bp = Blueprint(sch.name or "t", sch.description, list(sch.labels),
                  deflate_level=1)
        for t in sch.tiles:
            if t.logic is not None:
                bp.add_logic(t.logic.code, t.x, t.y, t.block, t.rotation,
                             [Link(l.name, l.x - t.x, l.y - t.y)
                              for l in t.logic.links])
            else:
                bp.add(t.block, t.x, t.y, t.rotation, t.config)
        bp.build()
        build_state = "ok"
    except Exception:
        build_state = "skip"
    return (logic_state, build_state, msgs)


def selftest(schem_dir: str | None = None, jobs: int | None = None) -> int:
    """
    Round-trip every blueprint found in the schematics folder.

    We measure two things separately, because they matter differently:
      * LOGIC fidelity -- every logic block's code + links survive a rebuild.
        This is what this tool is for, and it must be 100%.
      * BUILD success  -- the whole blueprint (including unrelated blocks such as
        power-node link lists) could be re-emitted. Nice to have, not required:
        we only ever *generate* processor blueprints.
    """
    import glob
    schem_dir = schem_dir or default_schematics_dir()
    if not schem_dir:
        print("selftest: could not find the Mindustry schematics folder.\n"
              "  set MINDUSTRY_SCHEMATICS=/path/to/schematics and retry.",
              file=sys.stderr)
        return 2
    print(f"selftest against {schem_dir}")
    files = sorted(glob.glob(os.path.join(schem_dir, "*.msch")))

    # One blueprint is ~85 ms of pure python and there are ~430 of them: this is
    # the slowest single step of gate.py --full.  Spread it over a pool when
    # asked to; the worker is module-level so it pickles, and any pool trouble
    # falls back to the serial loop.
    workers = jobs if jobs else min(8, os.cpu_count() or 1)
    if workers > 1 and len(files) > 4:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_selftest_file, files, chunksize=8))
        except Exception as exc:
            print(f"selftest: parallel run failed ({type(exc).__name__}: {exc}); "
                  f"falling back to serial")
            results = [_selftest_file(f) for f in files]
    else:
        results = [_selftest_file(f) for f in files]

    logic_ok = logic_bad = 0
    build_ok = build_skip = 0
    bad_examples: list[str] = []
    for logic_state, build_state, msgs in results:
        bad_examples.extend(msgs)
        if logic_state == "skip":
            continue
        if logic_state == "ok":
            logic_ok += 1
        else:
            logic_bad += 1
        if build_state == "ok":
            build_ok += 1
        else:
            build_skip += 1

    total = logic_ok + logic_bad
    print(f"logic fidelity : {logic_ok}/{total} exact round-trip")
    print(f"full build     : {build_ok} ok, {build_skip} skipped "
          f"(non-logic blocks with configs we don't re-emit, e.g. power-node links)")
    for e in bad_examples[:20]:
        print("   !! " + e)
    return 1 if logic_bad else 0


def link_name_for(block: str, index: int = 1) -> str:
    """Reproduce LogicBlock.getLinkName() + the game's numbering.

    The game takes the block's internal name, and if it contains '-' it keeps
    the LAST segment (unless that segment is 'large' or a number, in which case
    it keeps the second-to-last). Then it appends a 1-based index.
        overdrive-dome   -> dome1
        mass-driver      -> driver1
        large-logic-display -> display1
    """
    name = block
    if "-" in name:
        parts = name.split("-")
        last = parts[-1]
        if len(parts) >= 2 and (last == "large" or last.replace(".", "").isdigit()):
            name = parts[-2]
        else:
            name = last
    return f"{name}{index}"


def parse_with_block(spec: str) -> tuple[str, int, int, int | None]:
    """`block@dx,dy` or `block@dx,dy=contentId` -> (block, dx, dy, content_id)."""
    body, _, cid_s = spec.partition("=")
    block, _, pos = body.rpartition("@")
    if not block or not pos:
        raise ValueError(f"--with-block {spec!r} must look like block@dx,dy[=id]")
    try:
        dx, dy = (int(v) for v in pos.split(","))
    except ValueError:
        raise ValueError(f"bad position in --with-block {spec!r}")
    cid = None
    if cid_s:
        try:
            cid = int(cid_s)
        except ValueError:
            raise ValueError(f"bad content id in --with-block {spec!r}")
    return block, dx, dy, cid


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="mschgen", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="mlog file (default: stdin)")
    ap.add_argument("--code", help="mlog code inline (overrides source)")
    ap.add_argument("-o", "--out", default="out.msch", help="output .msch path")
    ap.add_argument("-f", "--format", choices=("code", "file", "both"),
                    default="code",
                    help="code = print the paste-ready base64 (default); "
                         "file = write .msch; both = do both")
    ap.add_argument("--name", default="blueprint", help="blueprint name tag")
    ap.add_argument("--description", default="", help="blueprint description tag")
    ap.add_argument("--label", action="append", default=[],
                    help="schematic label, repeatable")
    ap.add_argument("--block", default="micro-processor", help="processor type")
    ap.add_argument("--at", default="0,0", help="processor tile position x,y")
    ap.add_argument("--rotation", type=int, default=0)
    ap.add_argument("--link", action="append", default=[],
                    help="link in the form name@dx,dy (relative to the processor), repeatable")
    ap.add_argument("--with-block", action="append", default=[],
                    help="also place a block so a link has a real target: "
                         "block@dx,dy (optionally block@dx,dy=contentId). "
                         "Its link name is derived from the game's rule "
                         "(last dash-segment + index).")
    ap.add_argument("--selftest", action="store_true",
                    help="round-trip test against the real schematics folder")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.code is not None:
        code = args.code
    elif args.source:
        with open(args.source, encoding="utf-8") as fh:
            code = fh.read()
    else:
        code = sys.stdin.read()

    try:
        px, py = (int(v) for v in args.at.split(","))
    except ValueError:
        print("--at must look like x,y", file=sys.stderr)
        return 2

    links = [parse_link(s) for s in args.link]

    bp = Blueprint(args.name, args.description, args.label)
    bp.add_logic(code, px, py, args.block, args.rotation, links)

    # optionally place the blocks those links point at
    placed: list[str] = []
    for spec in args.with_block:
        blk, dx, dy, cid = parse_with_block(spec)
        bp.add_block(blk, px + dx, py + dy, 0, None, cid)
        placed.append(f"{blk}@{px+dx},{py+dy}"
                      + (f" (content id {cid})" if cid is not None else ""))

    data = bp.build()

    # always verify before emitting
    back = msch.parse_bytes(data)
    got = back.logic_tiles()
    if not (got and got[0].logic and got[0].logic.code.strip() == code.strip()
            and back.name == args.name):
        print("VERIFY FAILED -- not emitting", file=sys.stderr)
        return 1

    stmts = len([l for l in code.split("\n") if l.strip()])
    w, h = bp.bounds()
    if args.format in ("file", "both"):
        bp.write(args.out)
        print(f"# wrote {args.out}  ({len(data)} bytes)", file=sys.stderr)
    print(f"# {args.block} @({px},{py})  {stmts} stmts  {w}x{h}  "
          f"links={[f'{l.name}@{l.dx},{l.dy}' for l in links] or '(none)'}",
          file=sys.stderr)
    for p in placed:
        print(f"# placed block: {p}", file=sys.stderr)
    for wmsg in bp.link_report():
        print(f"# NOTE: {wmsg}", file=sys.stderr)
    if args.format in ("code", "both"):
        print(base64.b64encode(data).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
