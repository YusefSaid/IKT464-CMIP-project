#!/usr/bin/env python3
"""Map which parts of each variable in a partial MATLAB v7.3 file are downloaded.

Looks up chunks one at a time (reading only the small index entries on the way),
prints a coarse availability map per variable, lists the downloaded time steps
for the first and last block of grid points, and decodes one downloaded chunk to
check that the values are real. Read-only and low memory (reads ~64 KB of data).

    .venv/bin/python src/probe_chunks.py data/raw/historical/EC-EARTH3.mat.part

Needs peek_partial.py in the same folder.
"""
import argparse
import math
import os
import sys

import numpy as np
import h5py

from peek_partial import SIZES, PartialFile, human, short

ROWS, COLS, SAMPLES = 12, 6, 400
LEGEND = ("  # downloaded   . stored beyond the downloaded part   "
          "? index not downloaded yet   (blank) never stored")


def status(dsid, coord, avail: int) -> str:
    try:
        info = dsid.get_chunk_info_by_coord(coord)
    except Exception:
        return "?"                        # index path to this chunk not downloaded
    if info.byte_offset is None or info.size == 0:
        return " "
    if info.byte_offset < 1024:           # points into the header: incomplete index node
        return "?"
    return "#" if info.byte_offset + info.size <= avail else "."


def runs(flags, step: int, total: int) -> str:
    """Contiguous runs of True flags, as element index ranges."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append(f"{start * step:,}-{min(i * step, total) - 1:,}")
            start = None
    return ", ".join(out) if out else "none"


def probe(dset, avail: int) -> None:
    T, P = dset.shape
    ct, cp = dset.chunks
    nT, nP = math.ceil(T / ct), math.ceil(P / cp)
    dsid = dset.id
    print(f"\n== {dset.name.lstrip('/')}   h5py shape (time?, point?) = {T:,} x {P:,}, "
          f"chunks {ct} x {cp}, {nT:,} x {nP:,} = {nT * nP:,} chunks")

    # 1) coarse map
    tbs = sorted({round(i * (nT - 1) / (ROWS - 1)) for i in range(ROWS)})
    pbs = sorted({round(j * (nP - 1) / (COLS - 1)) for j in range(COLS)})
    print("   axis-0 index \\ axis-1 index: " + "".join(f"{pb * cp:>9,}" for pb in pbs))
    first_ok = None
    for tb in tbs:
        cells = []
        for pb in pbs:
            s = status(dsid, (tb * ct, pb * cp), avail)
            cells.append(f"{s:>9}")
            if s == "#" and first_ok is None:
                first_ok = (tb, pb)
        print(f"   {tb * ct:>12,} ({100 * tb * ct / T:3.0f}%)  " + "".join(cells))

    # 2) full scan along axis 0 for the first and last block of axis 1
    for pb in (0, nP - 1):
        ok = [status(dsid, (tb * ct, pb * cp), avail) == "#" for tb in range(nT)]
        print(f"   axis-1 {pb * cp:,}-{min((pb + 1) * cp, P) - 1:,}: downloaded axis-0 "
              f"ranges: {runs(ok, ct, T)}  ({100 * sum(ok) / nT:.0f}%)")

    # 3) random sample over the whole array
    rng = np.random.default_rng(0)
    got = [status(dsid, (int(rng.integers(nT)) * ct, int(rng.integers(nP)) * cp), avail)
           for _ in range(SAMPLES)]
    print(f"   random sample of {SAMPLES} chunks: {100 * got.count('#') / SAMPLES:.0f}% "
          f"downloaded, {100 * got.count('?') / SAMPLES:.0f}% index not downloaded yet")

    # 4) decode one downloaded chunk (gzip checksums reject anything not real)
    if first_ok is None:
        print("   no downloaded chunk found in the map, nothing to decode")
        return
    t0, p0 = first_ok[0] * ct, first_ok[1] * cp
    try:
        block = dset[t0:t0 + ct, p0:p0 + cp].astype(np.float64)
    except Exception as e:
        print(f"   test read of chunk at ({t0:,}, {p0:,}) FAILED: {short(e)}")
        return
    fin = block[np.isfinite(block)]
    stats = (f"min {fin.min():.3g}, mean {fin.mean():.3g}, max {fin.max():.3g}"
             if fin.size else "all NaN")
    print(f"   test read of chunk at ({t0:,}, {p0:,}): OK, {block.size:,} values, "
          f"{100 * (1 - fin.size / block.size):.0f}% NaN, {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("--size", type=int, help="expected full size in bytes")
    args = ap.parse_args()
    size = args.size or next((v for k, v in SIZES.items()
                              if f"/{k}/" in os.path.abspath(args.path)), None)
    if not size:
        sys.exit("Could not guess the expected size from the path; pass --size.")

    pf = PartialFile(args.path, size)
    print(f"File: {args.path}   downloaded {human(pf.avail)} of {human(size)}")
    print(LEGEND)
    with h5py.File(pf, "r") as h5:
        for name in h5.keys():
            try:
                obj = h5[name]
            except Exception:
                continue
            if isinstance(obj, h5py.Dataset) and obj.chunks and obj.ndim == 2:
                try:
                    probe(obj, pf.avail)
                except Exception as e:
                    print(f"\n== {name}: probe failed: {short(e)}")
    pf.close()


if __name__ == "__main__":
    main()
