"""Find damaged parts of an EC-EARTH3.mat file: chunk-index nodes and data chunks.

Principle
    HDF5 finds each chunk through an index tree (a v1 B-tree) whose nodes start with
    the marker "TREE". If one node is damaged, HDF5 stops, so this script does not use
    HDF5's own lookup. Like scan_btree.py, it scans the raw bytes for all index nodes
    and then checks three things:
      1. every node a parent node points to is a valid node (else: a damaged node);
      2. each variable has all its chunks (87,656/248 x 142,868/66 = 766,410);
      3. every chunk decompresses (gzip has a checksum, so one wrong byte fails).
    Damaged nodes and chunks are listed with their byte position, so that
    repair_chunks.py can re-download exactly those bytes.

Usage (from the project root):
    .venv/bin/python src/scan_chunks.py ssp126
    .venv/bin/python src/scan_chunks.py historical --vars fp uwnd vwnd

Output: data/raw/<scenario>/bad_chunks.csv (header only if nothing is damaged)
"""
from __future__ import annotations

import argparse
import csv
import mmap
import os
import struct
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
T, P, CT, CP, ITEM = 87_656, 142_868, 248, 66, 4
VARS = ("fp", "hs", "uwnd", "vwnd")          # order of the variables in the file
N_CHUNKS = -(-T // CT) * -(-P // CP)         # 766,410 per variable
RAW_CHUNK = CT * CP * ITEM
HDR = struct.Struct("<4sBBHQQ")              # "TREE", node type, level, entries, left, right
KEY = struct.Struct("<IIQQQ")                # chunk bytes, filter mask, offset t, offset p, 0
UNDEF = 0xFFFF_FFFF_FFFF_FFFF
FIELDS = ["kind", "var", "time_start", "point_start", "byte_offset", "nbytes", "zero_frac", "error"]
SEGMENT = 1 << 30                            # bytes per scanning task


def parse(mm, pos, size, file_size=None):
    """A v1 B-tree node for 2-D chunked data at pos in mm (of length size), or None.

    file_size bounds the child addresses (defaults to size, i.e. mm is the whole file)."""
    file_size = file_size or size
    if pos + HDR.size > size:
        return None
    sig, typ, level, n, left, right = HDR.unpack_from(mm, pos)
    if sig != b"TREE" or typ != 1 or level > 8 or not 1 <= n <= 1024:
        return None
    if pos + HDR.size + n * (KEY.size + 8) + KEY.size > size:
        return None
    q, ent = pos + HDR.size, []
    for _ in range(n):
        csize, mask, t, p, z = KEY.unpack_from(mm, q)
        (addr,) = struct.unpack_from("<Q", mm, q + KEY.size)
        q += KEY.size + 8
        if z or t % CT or p % CP or t >= T or p >= P or not 0 < addr < file_size:
            return None
        if level == 0 and not 0 < csize <= RAW_CHUNK + 4096:
            return None
        ent.append((t, p, addr, csize, mask))
    return level, left, right, ent


def _find_nodes(args):
    path, a, b = args
    out = []
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        size = len(mm)
        pos = mm.find(b"TREE", a, min(b + 3, size))
        while pos != -1:
            nd = parse(mm, pos, size)
            if nd:
                out.append((pos, *nd))
            pos = mm.find(b"TREE", pos + 1, min(b + 3, size))
    return out


def _check_chunks(args):
    path, recs = args                              # recs: (var, t, p, abs_addr, size, mask)
    bad = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for var, t, p, addr, size, mask in recs:
            if mask:
                continue
            raw = os.pread(fd, size, addr)
            try:
                n = len(zlib.decompress(raw))
                if n != RAW_CHUNK:
                    raise ValueError(f"decompressed to {n} bytes, expected {RAW_CHUNK}")
            except Exception as e:
                bad.append(dict(kind="chunk", var=var, time_start=t, point_start=p, byte_offset=addr,
                                nbytes=size, zero_frac=round(raw.count(0) / max(len(raw), 1), 3),
                                error=repr(e)[:120]))
    finally:
        os.close(fd)
    return len(recs), bad


def superblock(path):
    with open(path, "rb") as f:
        head = f.read(8193)
    for off in (0, 512, 1024, 2048, 4096, 8192):
        if head[off:off + 8] == b"\x89HDF\r\n\x1a\n":
            return off
    raise SystemExit(f"{path}: no HDF5 signature found")


def scan(scenario, variables, workers):
    path = str(RAW / scenario / "EC-EARTH3.mat")
    size, base, t0 = os.path.getsize(path), superblock(path), time.time()
    tasks = [(path, a, min(a + SEGMENT, size)) for a in range(0, size, SEGMENT)]
    nodes = {}
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_find_nodes, tasks), 1):
            for pos, level, left, right, ent in part:
                nodes[pos] = (level, left, right, ent)
            if i % 10 == 0 or i == len(tasks):
                print(f"  index scan {100 * i / len(tasks):5.1f}%  "
                      f"({(time.time() - t0) / 60:.1f} min, {len(nodes):,} nodes)", flush=True)

    # --- group the nodes into trees (one per variable) via parent-child and sibling links
    parent = {a: a for a in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bad, max_n = [], max(len(nd[3]) for nd in nodes.values())
    node_bytes = HDR.size + max_n * (KEY.size + 8) + KEY.size
    for pos, (level, left, right, ent) in nodes.items():
        for nb in (left, right):
            if nb != UNDEF and nb + base in nodes:
                parent[find(pos)] = find(nb + base)
        if level > 0:
            for t, p, addr, _, _ in ent:
                child = addr + base
                if child in nodes and nodes[child][0] == level - 1:
                    parent[find(pos)] = find(child)
    groups = {}
    for pos in nodes:
        groups.setdefault(find(pos), []).append(pos)
    # the four big trees, in file order, are fp, hs, uwnd, vwnd
    big = sorted(sorted(groups.values(), key=len)[-len(VARS):], key=min)
    label = {pos: v for v, g in zip(VARS, big) for pos in g}

    # --- 1. damaged nodes: a parent points to something that is not a valid node
    for pos, (level, left, right, ent) in nodes.items():
        if level == 0:
            continue
        for t, p, addr, _, _ in ent:
            child = addr + base
            if child not in nodes or nodes[child][0] != level - 1:
                with open(path, "rb") as f:
                    f.seek(child)
                    raw = f.read(node_bytes)
                bad.append(dict(kind="node", var=label.get(pos, "?"), time_start=t, point_start=p,
                                byte_offset=child, nbytes=node_bytes,
                                zero_frac=round(raw.count(0) / max(len(raw), 1), 3),
                                error=f"no valid level-{level - 1} index node here "
                                      f"(starts {raw[:4]!r})"))

    # --- 2. coverage per variable, 3. decompress every chunk
    recs = {v: {} for v in VARS}
    for pos, (level, left, right, ent) in nodes.items():
        v = label.get(pos)
        if level == 0 and v:
            for t, p, addr, csize, mask in ent:
                recs[v][(t, p)] = (v, t, p, addr + base, csize, mask)
    for v in VARS:
        print(f"  {v}: {len(recs[v]):,} of {N_CHUNKS:,} chunks found in the index"
              + ("" if len(recs[v]) == N_CHUNKS else "  <-- INCOMPLETE (damaged index nodes)"),
              flush=True)
    todo = sorted((r for v in variables for r in recs[v].values()), key=lambda r: r[3])
    step = max(1, -(-len(todo) // (workers * 16)))
    n_done = 0
    with Pool(workers) as pool:
        for n, b in pool.imap_unordered(_check_chunks, [(path, todo[i:i + step])
                                                       for i in range(0, len(todo), step)]):
            n_done += n
            bad += b
    print(f"  {n_done:,} chunks decompressed ({', '.join(variables)}), "
          f"{sum(r['kind'] == 'chunk' for r in bad)} damaged; "
          f"{(time.time() - t0) / 60:.1f} min in total", flush=True)

    bad.sort(key=lambda r: r["byte_offset"])
    out = RAW / scenario / "bad_chunks.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(bad)
    print(f"{scenario}: {sum(r['kind'] == 'node' for r in bad)} damaged index nodes, "
          f"{sum(r['kind'] == 'chunk' for r in bad)} damaged chunks -> {out}", flush=True)
    for r in bad[:15]:
        print(f"  {r['kind']:5s} {r['var']:4s} steps {r['time_start']:,}+ points {r['point_start']:,}+ "
              f"at byte {r['byte_offset']:,} ({r['nbytes']:,} B, {100 * r['zero_frac']:.0f}% zeros)",
              flush=True)
    if any(r["kind"] == "node" for r in bad):
        print("NOTE: chunks listed in damaged index nodes could not be checked yet; "
              "after repair_chunks.py, run this scan again.", flush=True)
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="List damaged index nodes and chunks.")
    ap.add_argument("scenarios", nargs="+", choices=["historical", "ssp126", "ssp585"])
    ap.add_argument("--vars", nargs="+", default=list(VARS), choices=VARS,
                    help="variables whose chunks are decompressed (the index is always checked)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    for s in a.scenarios:
        scan(s, a.vars, a.workers)
