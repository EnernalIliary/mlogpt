"""
msch.py -- Mindustry blueprint (.msch) parser that extracts logic source code.

Format derived from Mindustry-Source (build 159.1):

  Schematics.read()                      core/src/mindustry/game/Schematics.java:543
  Schematics.write()                     core/src/mindustry/game/Schematics.java:623
  LogicBlock.compress()                  core/src/mindustry/world/blocks/logic/LogicBlock.java:141
  LogicBlock.LogicBuild.readCompressed() core/src/mindustry/world/blocks/logic/LogicBlock.java:279
  TypeIO.writeObject()/readObject()      core/src/mindustry/io/TypeIO.java:42,178

.msch layout
------------
  'm' 's' 'c' 'h'         4-byte ASCII magic
  version                 1 byte  (current version = 1)
  ---- zlib (Deflater/Inflater) stream ----
  width, height           int16, int16
  tagCount                uint8
  tagCount * (UTF key, UTF value)          <- "labels" / "contentMap" / "name" / "description"
  blockDictLen            uint8
  blockDictLen * UTF block name
  totalTiles              int32
  totalTiles * {
      blockIndex          int8
      position            int32 (Point2.pack)
      config              TypeIO object   <- logic lives here
      rotation            int8
  }

Logic block config
------------------
LogicBlock declares config(byte[].class, ...) so the tile config is a TypeIO
type-14 byte[] holding a *second* zlib stream:

  logicVersion            uint8   (0 = legacy links-only, 1 = current)
  byteLen                 int32
  byteLen * bytes         <- UTF-8 of the logic source: ONE ASSEMBLY STATEMENT PER LINE
  linkCount               int32
  linkCount * { UTF name, int16 x, int16 y }

Because `bytes` is exactly the text produced by LAssembler.write(), the original
logic is recoverable verbatim -- no decompilation required.
"""

from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass, field

MAGIC = b"msch"
CURRENT_VERSION = 1

# TypeIO object type tags -- TypeIO.java:42 (write) / :178 (read)
T_NULL = 0
T_INT = 1
T_LONG = 2
T_FLOAT = 3
T_STRING = 4
T_CONTENT = 5
T_INTARRAY = 6
T_POINT = 7
T_POINTARRAY = 8
T_TECHNODE = 9
T_BOOL = 10
T_DOUBLE = 11
T_BUILDING = 12
T_LACCESS = 13
T_BYTEARRAY = 14
T_UNITCOMMAND = 15
T_BOOLARRAY = 16
T_UNIT = 17
T_VEC2ARRAY = 18
T_VEC2 = 19
T_TEAM = 20
T_INTS = 21
T_OBJECTARRAY = 22
T_UNITCOMMANDBOX = 23

# ContentType ordinals, matching ContentType.all[] index order in Mindustry.
CONTENT_TYPES = [
    "item", "block", "mech_UNUSED", "bullet", "liquid", "status",
    "unit", "weather", "effect", "sector", "loadout_UNUSED", "typeid_UNUSED",
    "error", "planet", "ammo_UNUSED", "team",
]


class MschError(Exception):
    """Raised when a blueprint cannot be parsed."""


class Reader:
    """Big-endian cursor over an in-memory buffer (mirrors arc's Reads)."""

    def __init__(self, data: bytes, offset: int = 0, limit: int | None = None):
        self.data = data
        self.pos = offset
        self.limit = len(data) if limit is None else limit

    def remaining(self) -> int:
        return self.limit - self.pos

    def _take(self, n: int) -> bytes:
        if self.pos + n > self.limit:
            raise MschError(
                f"unexpected end of data: wanted {n} bytes at offset {self.pos}, "
                f"only {self.remaining()} left"
            )
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def b(self) -> int:
        return struct.unpack(">b", self._take(1))[0]

    def ub(self) -> int:
        return self._take(1)[0]

    def boolean(self) -> bool:
        return self._take(1)[0] != 0

    def s(self) -> int:
        return struct.unpack(">h", self._take(2))[0]

    def us(self) -> int:
        return struct.unpack(">H", self._take(2))[0]

    def i(self) -> int:
        return struct.unpack(">i", self._take(4))[0]

    def ui(self) -> int:
        return struct.unpack(">I", self._take(4))[0]

    def l(self) -> int:
        return struct.unpack(">q", self._take(8))[0]

    def f(self) -> float:
        return struct.unpack(">f", self._take(4))[0]

    def d(self) -> float:
        return struct.unpack(">d", self._take(8))[0]

    def utf(self) -> str:
        """Java DataInputStream.readUTF: uint16 length + modified UTF-8."""
        n = self.us()
        raw = self._take(n)
        # Java modified UTF-8: CESU-8 style, encode NUL as 0xC0 0x80.
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - decode with replace never raises
            return raw.decode("latin-1")

    def bytes(self, n: int) -> bytes:
        return self._take(n)


@dataclass
class LogicLink:
    """A named link from a processor to another building (LogicLink in source)."""
    name: str
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.name}@({self.x},{self.y})"


@dataclass
class LogicBlockData:
    """Decoded logic payload of one processor/display/memory block."""
    version: int
    code: str
    links: list[LogicLink] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        return self.code.split("\n") if self.code else []

    def statement_lines(self) -> list[str]:
        """Non-empty, stripped lines (what the assembly editor shows)."""
        return [ln.strip() for ln in self.lines if ln.strip()]


@dataclass
class Stile:
    """One placed building (Schematic.Stile in source)."""
    block: str
    x: int
    y: int
    rotation: int
    config: object = None
    raw_config: bytes | None = None
    logic: LogicBlockData | None = None


@dataclass
class Schematic:
    width: int
    height: int
    tags: dict[str, str]
    tiles: list[Stile]
    version: int = CURRENT_VERSION
    labels: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tags.get("name", "unknown")

    @property
    def description(self) -> str:
        return self.tags.get("description", "")

    def logic_tiles(self) -> list[Stile]:
        return [t for t in self.tiles if t.logic is not None]

    def block_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.tiles:
            counts[t.block] = counts.get(t.block, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _read_object(r: Reader, depth: int = 0) -> object:
    """Minimal but faithful TypeIO.readObject (box=False, safe=False)."""
    if depth > 32:
        raise MschError("config object nested too deeply")

    tag = r.b()
    if tag == T_NULL:
        return None
    if tag == T_INT:
        return r.i()
    if tag == T_LONG:
        return r.l()
    if tag == T_FLOAT:
        return r.f()
    if tag == T_STRING:
        return r.utf() if r.b() != 0 else None
    if tag == T_CONTENT:
        ctype = r.ub()
        cid = r.s()
        name = CONTENT_TYPES[ctype] if ctype < len(CONTENT_TYPES) else f"type{ctype}"
        return {"__content__": name, "id": cid}
    if tag == T_INTARRAY:
        n = r.s()
        _check_array(n)
        return [r.i() for _ in range(n)]
    if tag == T_POINT:
        return {"__point__": (r.i(), r.i())}
    if tag == T_POINTARRAY:
        n = r.ub()
        out = []
        for _ in range(n):
            packed = r.i()
            out.append(_unpack_point(packed))
        return out
    if tag == T_TECHNODE:
        return {"__technode__": (r.ub(), r.s())}
    if tag == T_BOOL:
        return r.boolean()
    if tag == T_DOUBLE:
        return r.d()
    if tag == T_BUILDING:
        return {"__building__": r.i()}
    if tag == T_LACCESS:
        return {"__laccess__": r.s()}
    if tag == T_BYTEARRAY:
        n = r.i()
        _check_array(n, kind="byte")
        return r.bytes(n)
    if tag == T_UNITCOMMAND:
        r.b()
        return None
    if tag == T_BOOLARRAY:
        n = r.i()
        _check_array(n)
        return [r.boolean() for _ in range(n)]
    if tag == T_UNIT:
        return {"__unit__": r.i()}
    if tag == T_VEC2ARRAY:
        n = r.s()
        _check_array(n)
        return [{"__vec2__": (r.f(), r.f())} for _ in range(n)]
    if tag == T_VEC2:
        return {"__vec2__": (r.f(), r.f())}
    if tag == T_TEAM:
        return {"__team__": r.ub()}
    if tag == T_INTS:
        n = r.us()
        _check_array(n)
        return [r.i() for _ in range(n)]
    if tag == T_OBJECTARRAY:
        n = r.i()
        _check_array(n)
        return [_read_object(r, depth + 1) for _ in range(n)]
    if tag == T_UNITCOMMANDBOX:
        return {"__unitcommand__": r.us()}
    raise MschError(f"unknown TypeIO object type {tag} at offset {r.pos - 1}")


def _check_array(n: int, kind: str = "array") -> None:
    if n < 0 or n > 1_000_000:
        raise MschError(f"invalid {kind} size {n}")


def _unpack_point(packed: int) -> tuple[int, int]:
    """Point2.unpack: x = (short)(packed >> 16), y = (short)(packed & 0xFFFF)."""
    x = (packed >> 16) & 0xFFFF
    y = packed & 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    if y >= 0x8000:
        y -= 0x10000
    return x, y


def _pack_point(x: int, y: int) -> int:
    return ((x & 0xFFFF) << 16) | (y & 0xFFFF)


def parse_logic(data: bytes, relative_to: tuple[int, int] | None = None) -> LogicBlockData:
    """Decode one LogicBlock config payload (LogicBuild.readCompressed)."""
    try:
        raw = zlib.decompress(data)
    except zlib.error as e:
        raise MschError(f"logic payload is not zlib: {e}") from e

    r = Reader(raw)
    version = r.ub()
    byte_len = r.i()
    if byte_len < 0 or byte_len > 100_000_000:
        raise MschError(f"malformed logic data, length {byte_len}")
    code_bytes = r.bytes(byte_len)
    code = code_bytes.decode("utf-8", errors="replace")

    links: list[LogicLink] = []
    total = r.i()
    if total < 0 or total > 100_000:
        raise MschError(f"malformed logic link count {total}")

    if version == 0:
        for _ in range(total):
            r.i()  # legacy absolute positions, ignored by the game too
    else:
        seen: set[tuple[int, int]] = set()
        for _ in range(total):
            name = r.utf()
            x, y = r.s(), r.s()
            if relative_to is not None:
                x += relative_to[0]
                y += relative_to[1]
            if (x, y) in seen:
                continue  # game drops duplicate links (usedBuildings set)
            seen.add((x, y))
            links.append(LogicLink(name, x, y))

    return LogicBlockData(version=version, code=code, links=links)


def parse_bytes(data: bytes) -> Schematic:
    """Parse raw .msch bytes into a Schematic."""
    if len(data) < 5:
        raise MschError("file too small to be a schematic")
    if data[:4] != MAGIC:
        raise MschError("not a schematic file (missing 'msch' header)")

    version = data[4]
    if version > CURRENT_VERSION:
        raise MschError(
            f"unknown schematic version {version}; this parser supports up to {CURRENT_VERSION}"
        )

    body = data[5:]
    try:
        raw = zlib.decompress(body) if version > 0 else zlib.decompress(body)
    except zlib.error as e:
        raise MschError(f"body is not a valid zlib stream: {e}") from e

    r = Reader(raw)
    width, height = r.s(), r.s()

    tags: dict[str, str] = {}
    for _ in range(r.ub()):
        # NOTE: must be two statements. Python evaluates the RHS of a subscript
        # assignment BEFORE the subscript key, so `tags[r.utf()] = r.utf()` would
        # read the value first and store the pair inverted.
        key = r.utf()
        tags[key] = r.utf()

    labels: list[str] = []
    if "labels" in tags:
        import json
        try:
            parsed = json.loads(tags["labels"])
            if isinstance(parsed, list):
                labels = [str(x) for x in parsed]
        except Exception:
            pass

    block_dict: list[str] = [r.utf() for _ in range(r.ub())]
    total = r.i()
    if total < 0 or total > 1_000_000:
        raise MschError(f"invalid schematic: too many blocks ({total})")

    tiles: list[Stile] = []
    for _ in range(total):
        bidx = r.b()
        if not (0 <= bidx < len(block_dict)):
            raise MschError(f"block index {bidx} out of range (dict has {len(block_dict)})")
        block = block_dict[bidx]
        pos = r.i()
        px, py = _unpack_point(pos)
        config = _read_object(r)
        rotation = r.b()

        if block == "air":
            continue

        tile = Stile(block=block, x=px, y=py, rotation=rotation, config=config)

        # LogicBlock declares config(byte[].class, ...); a byte[] config on a
        # logic block is the deflated {version, code, links} record.
        if isinstance(config, (bytes, bytearray)):
            try:
                tile.logic = parse_logic(bytes(config), relative_to=(px, py))
                tile.raw_config = bytes(config)
            except MschError:
                tile.raw_config = bytes(config)

        tiles.append(tile)

    return Schematic(width=width, height=height, tags=tags, tiles=tiles,
                     version=version, labels=labels)


def parse_file(path) -> Schematic:
    with open(path, "rb") as fh:
        return parse_bytes(fh.read())


def parse_base64(text: str) -> Schematic:
    """Parse a blueprint copied from the in-game clipboard."""
    return parse_bytes(base64.b64decode(text.strip()))


def build_logic_data(code: str, links: list[LogicLink]) -> bytes:
    """Inverse of parse_logic -- used by the verifier and by generators."""
    out = bytearray()
    out.append(1)
    code_bytes = code.encode("utf-8")
    out += struct.pack(">i", len(code_bytes))
    out += code_bytes
    out += struct.pack(">i", len(links))
    for link in links:
        nb = link.name.encode("utf-8")
        out += struct.pack(">H", len(nb)) + nb
        out += struct.pack(">hh", link.x, link.y)
    return zlib.compress(bytes(out), 9)


def build_schematic(width: int, height: int, tags: dict[str, str],
                    tiles: list[Stile]) -> bytes:
    """Inverse of parse_bytes -- used to round-trip-test the parser."""
    body = bytearray()
    body += struct.pack(">hh", width, height)
    body.append(len(tags))
    for k, v in tags.items():
        for s in (k, v):
            sb = s.encode("utf-8")
            body += struct.pack(">H", len(sb)) + sb

    names: list[str] = []
    for t in tiles:
        if t.block not in names:
            names.append(t.block)
    body.append(len(names))
    for n in names:
        nb = n.encode("utf-8")
        body += struct.pack(">H", len(nb)) + nb

    body += struct.pack(">i", len(tiles))
    for t in tiles:
        body.append(names.index(t.block))
        body += struct.pack(">i", _pack_point(t.x, t.y))
        _write_object(body, t.config)
        body.append(t.rotation & 0xFF)

    return MAGIC + bytes([CURRENT_VERSION]) + zlib.compress(bytes(body), 9)


def _write_object(out: bytearray, obj: object) -> None:
    """Minimal TypeIO.writeObject covering the types seen in blueprints."""
    if obj is None:
        out.append(T_NULL)
    elif isinstance(obj, bool):
        out.append(T_BOOL)
        out.append(1 if obj else 0)
    elif isinstance(obj, int):
        out.append(T_INT)
        out += struct.pack(">i", obj)
    elif isinstance(obj, float):
        out.append(T_FLOAT)
        out += struct.pack(">f", obj)
    elif isinstance(obj, str):
        out.append(T_STRING)
        out.append(1)
        sb = obj.encode("utf-8")
        out += struct.pack(">H", len(sb)) + sb
    elif isinstance(obj, (bytes, bytearray)):
        out.append(T_BYTEARRAY)
        out += struct.pack(">i", len(obj))
        out += bytes(obj)
    elif isinstance(obj, dict) and "__content__" in obj:
        out.append(T_CONTENT)
        out.append(CONTENT_TYPES.index(obj["__content__"]))
        out += struct.pack(">h", obj["id"])
    elif isinstance(obj, dict) and "__point__" in obj:
        out.append(T_POINT)
        out += struct.pack(">ii", *obj["__point__"])
    else:
        raise MschError(f"cannot serialize config of type {type(obj).__name__}")


def decode_variant(text: str) -> Schematic:
    """Accept either base64 clipboard text or raw binary bytes-as-str."""
    stripped = "".join(text.split())
    try:
        return parse_base64(stripped)
    except Exception:
        return parse_bytes(text.encode("latin-1"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    sch = parse_file(sys.argv[1])
    print(f"name      : {sch.name}")
    print(f"size      : {sch.width}x{sch.height}")
    print(f"tiles     : {len(sch.tiles)}")
    print(f"labels    : {sch.labels}")
    print(f"tags      : {list(sch.tags)}")
    for t in sch.logic_tiles():
        assert t.logic is not None
        print(f"\n--- {t.block} at ({t.x},{t.y}) "
              f"v{t.logic.version}, {len(t.logic.statement_lines())} statements, "
              f"{len(t.logic.links)} links ---")
        print(t.logic.code)
