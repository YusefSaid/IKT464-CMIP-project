#!/usr/bin/env python3
"""Find chunk-index pieces inside a partial MATLAB v7.3 (HDF5) file.

HDF5 finds each chunk of data through an index tree (a v1 B-tree). In the
partial historical file every normal lookup fails, because nodes near the top
of the tree are beyond the downloaded part. The bottom-level nodes (leaves),
which record where each chunk sits, may still be in the downloaded part. This
script scans the raw bytes for them (they start with the marker "TREE") and
reports:
  - how many leaves and chunk records it finds, and how many of those chunks
    are fully downloaded;
  - groups of leaves linked to each other (each group belongs to one variable),
    with the time-step / grid-point range they cover, and the values of one
    decoded chunk so we can tell which variable it is (hs, fp, uwnd or vwnd).

Read-only (memory-maps the file). Takes a minute or a few for ~10 GB.

    .venv/bin/python src/scan_btree.py data/raw/historical/EC-EARTH3.mat.part

Saves the locations of all downloaded chunks to <path>.chunks.npz, so a reader
can use them without scanning again.
"""
import argparse
import mmap
import os
import struct
import sys
import zlib

import numpy as np

T, P, CT, CP, ITEM = 87_656, 142_868, 248, 66, 4       # from peek_partial.py
SIZES = {"historical": 59_982_783_406, "ssp126": 60_656_201_821,
         "ssp585": 60_874_380_344}
HDR = struct.Struct("<4sBBHQQ")   # "TREE", node type, level, entries, left, right
KEY = struct.Struct("<IIQQQ")     # chunk bytes, filter mask, offset t, offset p, 0
CHILD = struct.Struct("<Q")
UNDEF = 0xFFFF_FFFF_FFFF_FFFF
RAW_CHUNK = CT * CP * ITEM


def parse(buf, pos: int, avail: int, full: int):
    """Parse a v1 B-tree node for 2-D chunked data at pos, or return None."""
    if pos + HDR.size > avail:
        return None
    _, typ, level, n, left, right = HDR.unpack_from(buf, pos)
    if typ != 1 or level > 8 or not 1 <= n <= 1024:
        return None
    if pos + HDR.size + n * (KEY.size + CHILD.size) + KEY.size > avail:
        return None
    q, entries = pos + HDR.size, []
    for _ in range(n):
        size, mask, t, p, z = KEY.unpack_from(buf, q)
        (addr,) = CHILD.unpack_from(buf, q + KEY.size)
        q += KEY.size + CHILD.size
        if z or t % CT or p % CP or t >= T or p >= P or not 0 < addr < full:
            return None
        if level == 0 and not 0 < size <= RAW_CHUNK + 4096:
            return None
        entries.append((t, p, addr, size, mask))
    return {"pos": pos, "level": level, "left": left, "right": right,
            "entries": entries}


def superblock_offset(mm) -> int:
    """Where the HDF5 data starts; MATLAB v7.3 files have a 512-byte header first."""
    for off in (0, 512, 1024, 2048, 4096, 8192):
        if mm[off:off + 8] == b"\x89HDF\r\n\x1a\n":
            return off
    return 0


def pick_base(nodes: dict, candidates) -> tuple:
    """HDF5 addresses are relative to a base; pick the one that links the leaves."""
    best = (0, -1)
    for base in dict.fromkeys(candidates):
        hits = sum(1 for nd in nodes.values()
                   for b in (nd["left"], nd["right"])
                   if b != UNDEF and b + base in nodes)
        if hits > best[1]:
            best = (base, hits)
    return best


def scan(mm, avail: int, full: int) -> dict:
    nodes, pos, next_report = {}, mm.find(b"TREE"), 1 << 30
    while pos != -1:
        if pos >= next_report:
            print(f"  scanned {pos / 2**30:.0f} GiB, {len(nodes):,} nodes so far",
                  file=sys.stderr, flush=True)
            next_report += 1 << 30
        node = parse(mm, pos, avail, full)
        if node:
            nodes[pos] = node
        pos = mm.find(b"TREE", pos + 1)
    return nodes


def chains(leaves: dict, base: int) -> list:
    """Group leaves connected through their left/right sibling links."""
    parent = {a: a for a in leaves}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, nd in leaves.items():
        for b in (nd["left"], nd["right"]):
            if b != UNDEF and b + base in leaves:
                parent[find(a)] = find(b + base)
    groups = {}
    for a in leaves:
        groups.setdefault(find(a), []).append(leaves[a])
    return sorted(groups.values(), key=lambda g: min(nd["pos"] for nd in g))


def decode(mm, addr: int, size: int, shuffle: bool) -> np.ndarray:
    raw = zlib.decompress(mm[addr:addr + size])
    if len(raw) != RAW_CHUNK:
        raise ValueError(f"chunk decoded to {len(raw)} bytes, expected {RAW_CHUNK}")
    if shuffle:
        raw = np.frombuffer(raw, np.uint8).reshape(ITEM, -1).T.tobytes()
    return np.frombuffer(raw, "<f4").reshape(CT, CP)


def uses_shuffle(path: str, full: int):
    try:
        import h5py
        from peek_partial import PartialFile
        pf = PartialFile(path, full)
        with h5py.File(pf, "r") as h5:
            s = {bool(h5[k].shuffle) for k in ("hs", "fp", "uwnd", "vwnd") if k in h5}
        pf.close()
        return s.pop() if len(s) == 1 else None
    except Exception:
        return None


def runs(flags, step: int, total: int) -> str:
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append(f"{start * step:,}-{min(i * step, total) - 1:,}")
            start = None
    return ", ".join(out) if out else "none"


def report(nodes: dict, mm, avail: int, base: int, shuffle: bool, save: str,
           top: int = 12) -> None:
    nT, nP = -(-T // CT), -(-P // CP)
    by_level = {}
    for nd in nodes.values():
        by_level[nd["level"]] = by_level.get(nd["level"], 0) + 1
    print("index nodes found by level (0 = leaves): "
          + (", ".join(f"level {k}: {v:,}" for k, v in sorted(by_level.items())) or "none"))
    leaves = {a: nd for a, nd in nodes.items() if nd["level"] == 0}
    if not leaves:
        print("No leaf nodes in the downloaded part, so chunk locations can't be "
              "recovered from it.")
        return
    recs = [e for nd in leaves.values() for e in nd["entries"]]
    ok = sum(1 for _, _, a, s, _ in recs if a + base + s <= avail)
    print(f"chunk records in those leaves: {len(recs):,}, of which {ok:,} have their "
          f"data fully downloaded (one variable has {nT * nP:,} chunks)")
    masked = sum(1 for e in recs if e[4])
    if masked:
        print(f"note: {masked:,} records have a non-zero filter mask (skipped)")

    groups = chains(leaves, base)
    groups.sort(key=lambda g: min(nd["pos"] for nd in g))
    print(f"\n{len(groups):,} linked group(s) of leaves (one per variable if all is well); "
          f"the {min(top, len(groups))} largest, in file order:")
    big = sorted(range(len(groups)),
                 key=lambda i: -sum(len(nd["entries"]) for nd in groups[i]))[:top]
    cols = {k: [] for k in ("group", "t", "p", "addr", "size")}
    for gi, g in enumerate(groups):
        good = [e for nd in g for e in nd["entries"]
                if e[2] + base + e[3] <= avail and not e[4]]
        for t, p, a, s, _ in good:
            cols["group"].append(gi); cols["t"].append(t); cols["p"].append(p)
            cols["addr"].append(a + base); cols["size"].append(s)
        if gi not in big:
            continue
        es = [e for nd in g for e in nd["entries"]]
        lo = min(nd["pos"] for nd in g) / 2**30
        hi = max(nd["pos"] for nd in g) / 2**30
        print(f"  group {gi}: {len(g):,} leaves at {lo:.2f}-{hi:.2f} GiB, {len(es):,} chunks "
              f"({len(good):,} downloaded)")
        per_tb = {}
        for t, p, *_ in good:
            per_tb.setdefault(t // CT, set()).add(p)
        complete = [len(per_tb.get(k, ())) == nP for k in range(nT)]
        print(f"      time steps complete for ALL grid points: {runs(complete, CT, T)} "
              f"({sum(complete)} of {nT} blocks)")
        if good:
            t, p, a, s, _ = min(good, key=lambda e: (e[0], e[1]))
            try:
                v = decode(mm, a + base, s, shuffle).astype(np.float64)
                f = v[np.isfinite(v)]
                st = (f"min {f.min():.3g}, mean {f.mean():.3g}, max {f.max():.3g}"
                      if f.size else "all NaN")
                print(f"      decoded chunk at t={t:,}, p={p:,}: "
                      f"{100 * (1 - f.size / v.size):.0f}% NaN, {st}")
            except Exception as e:
                print(f"      decoding chunk at t={t:,}, p={p:,} failed: {e}")
    if save and cols["t"]:
        np.savez_compressed(save, **{k: np.asarray(v, np.int64) for k, v in cols.items()},
                            shape=np.array([T, P]), chunks=np.array([CT, CP]))
        print(f"\nsaved index of {len(cols['t']):,} downloaded chunks to {save}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--size", type=int, help="expected full size in bytes")
    ap.add_argument("--save", default=None,
                    help="where to save the chunk index (default: <path>.chunks.npz)")
    args = ap.parse_args()
    save = args.save or args.path + ".chunks.npz"
    full = args.size or next((v for k, v in SIZES.items()
                              if f"/{k}/" in os.path.abspath(args.path)), None)
    if not full:
        sys.exit("Could not guess the expected size from the path; pass --size.")

    shuffle = uses_shuffle(args.path, full)
    print(f"shuffle filter: {shuffle if shuffle is not None else 'unknown, assuming no'}")
    with open(args.path, "rb") as f:
        avail = os.fstat(f.fileno()).st_size
        print(f"scanning {avail / 2**30:.1f} GiB ...", flush=True)
        with mmap.mmap(f.fileno(), avail, access=mmap.ACCESS_READ) as mm:
            sb = superblock_offset(mm)
            nodes = scan(mm, avail, full)
            base, hits = pick_base(nodes, (sb, 0))
            print(f"HDF5 data starts at byte {sb}; addresses offset by {base} "
                  f"({hits:,} sibling links resolve)")
            report(nodes, mm, avail, base, bool(shuffle), save)


if __name__ == "__main__":
    main()