#!/usr/bin/env python3
"""Read EC-EARTH3 data from a partially downloaded .mat file.

Uses the chunk index saved by scan_btree.py (<file>.chunks.npz). Read-only; it
only decodes the chunks you ask for, so memory use matches the size of the result.

Layout (from the file): each variable is a (time step, sea grid point) array of
87,656 x 142,868 float32 values. The time axis is probably 3-hourly from
1985-01-01 (87,656 = 30 years x 8 per day), and the variable names below are
inferred from value ranges and file order. Confirm both with Francesco or the
file's metadata before relying on them in the report.

Anything not downloaded yet raises MissingData; it never comes back as zeros.

    import sys; sys.path.insert(0, "src")
    from partial_reader import PartialMat
    with PartialMat("data/raw/historical/EC-EARTH3.mat.part") as m:
        n = m.available_steps()                  # steps 0..n-1 complete everywhere
        hs = m.read("hs", points=[1000, 2000], steps=slice(0, n))   # (time, points)

When the download finishes, read the complete file with h5py instead; the same
indexing works:  h5py.File(path)["hs"][0:n, [1000, 2000]]
"""
import os
import zlib

import numpy as np

VARS = {"fp": 0, "hs": 1, "uwnd": 2, "vwnd": 3}   # group order in the file


class MissingData(LookupError):
    """The requested data is not in the downloaded part yet."""


class PartialMat:
    def __init__(self, path: str, index: str | None = None):
        z = np.load(index or path + ".chunks.npz")
        self.T, self.P = (int(x) for x in z["shape"])
        self.ct, self.cp = (int(x) for x in z["chunks"])
        self.nT, self.nP = -(-self.T // self.ct), -(-self.P // self.cp)
        groups = int(z["group"].max()) + 1
        if groups != len(VARS):
            raise ValueError(f"index has {groups} groups, expected {len(VARS)}; "
                             "re-run scan_btree.py")
        self._addr = np.full((groups, self.nT, self.nP), -1, np.int64)
        self._size = np.zeros((groups, self.nT, self.nP), np.int64)
        g, tb, pb = z["group"], z["t"] // self.ct, z["p"] // self.cp
        self._addr[g, tb, pb] = z["addr"]
        self._size[g, tb, pb] = z["size"]
        self._fd = os.open(path, os.O_RDONLY)

    # -- context manager ---------------------------------------------------
    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- what is available -------------------------------------------------
    def complete_blocks(self, var: str) -> np.ndarray:
        """Boolean per time block: downloaded for every grid point."""
        return (self._addr[VARS[var]] >= 0).all(axis=1)

    def available_steps(self, var: str | None = None) -> int:
        """Number of leading time steps complete for all points (all vars if None)."""
        names = [var] if var else list(VARS)
        n = self.nT
        for v in names:
            done = self.complete_blocks(v)
            n = min(n, int(np.argmin(done)) if not done.all() else self.nT)
        return min(n * self.ct, self.T)

    # -- reading -----------------------------------------------------------
    def _chunk(self, g: int, tb: int, pb: int, var: str) -> np.ndarray:
        addr = int(self._addr[g, tb, pb])
        if addr < 0:
            raise MissingData(
                f"{var}: time steps {tb * self.ct:,}-{(tb + 1) * self.ct - 1:,} at grid "
                f"points {pb * self.cp:,}-{(pb + 1) * self.cp - 1:,} are not downloaded yet")
        raw = os.pread(self._fd, int(self._size[g, tb, pb]), addr)
        return np.frombuffer(zlib.decompress(raw), "<f4").reshape(self.ct, self.cp)

    def read(self, var: str, points, steps=slice(None)) -> np.ndarray:
        """Return a float32 array of shape (len(steps), len(points))."""
        g = VARS[var]
        pts = np.atleast_1d(np.arange(self.P)[points] if isinstance(points, slice)
                            else np.asarray(points, dtype=np.int64))
        stp = np.arange(self.T)[steps] if isinstance(steps, slice) \
            else np.atleast_1d(np.asarray(steps, dtype=np.int64))
        if pts.size and (pts.min() < 0 or pts.max() >= self.P):
            raise IndexError(f"grid points must be in 0..{self.P - 1}")
        if stp.size and (stp.min() < 0 or stp.max() >= self.T):
            raise IndexError(f"time steps must be in 0..{self.T - 1}")
        out = np.empty((stp.size, pts.size), np.float32)
        tbs, pbs = stp // self.ct, pts // self.cp
        cols = {int(pb): np.nonzero(pbs == pb)[0] for pb in np.unique(pbs)}
        for tb in np.unique(tbs):
            rows = np.nonzero(tbs == tb)[0]
            for pb, c in cols.items():
                block = self._chunk(g, int(tb), pb, var)
                out[np.ix_(rows, c)] = block[np.ix_(stp[rows] - tb * self.ct,
                                                    pts[c] - pb * self.cp)]
        return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/historical/EC-EARTH3.mat.part"
    with PartialMat(path) as m:
        n = m.available_steps()
        print(f"{path}: steps 0-{n - 1:,} complete for all {m.P:,} points in all variables"
              f" (~{n / 8 / 365.25:.1f} years if 3-hourly)")
        pts = [0, m.P // 4, m.P // 2, 3 * m.P // 4, m.P - 1]
        for v in VARS:
            x = m.read(v, pts, slice(0, n))
            print(f"  {v:5s} at points {pts}: "
                  + ", ".join(f"mean {np.nanmean(x[:, i]):7.3f}" if np.isfinite(x[:, i]).any()
                              else "all NaN" for i in range(len(pts))))
