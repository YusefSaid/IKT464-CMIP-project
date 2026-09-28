"""Shared helpers for the IKT464 EC-EARTH3 notebooks.

Import in every notebook (see the setup cell in the notebook), so the loading and
saving code lives in one place.

    with Scenario("historical") as hist:
        n = hist.available_steps()
        hs = hist.read("hs", points=[1000, 2000], steps=slice(0, n))   # (time, points)
    save_fig(fig, "hs_timeseries", "eda")        # -> fig/eda/hs_timeseries.png
    grid = load_grid()                            # coordinates of every sea point
    p = nearest_point(grid, lat=45.0, lon=-30.0)  # point index for a location

Scenario reads from the partial download (data/raw/<scenario>/EC-EARTH3.mat.part,
via partial_reader.py) until the complete file EC-EARTH3.mat exists, then reads
the complete file with h5py. Notebook code does not change when a download
finishes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# -- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "fig"

SCENARIOS = ("historical", "ssp126", "ssp585")

# Inferred from value ranges and standard WaveWatch III names; confirm with Francesco.
VARIABLES = {
    "hs": ("significant wave height", "m"),
    "fp": ("peak frequency", "Hz"),
    "uwnd": ("eastward wind (assumed)", "m/s"),
    "vwnd": ("northward wind (assumed)", "m/s"),
}


# -- reading -------------------------------------------------------------------
class Scenario:
    """One scenario's data: complete .mat if downloaded, otherwise the partial file."""

    def __init__(self, name: str):
        if name not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.name = name
        full = RAW / name / "EC-EARTH3.mat"
        self._h5 = self._partial = None
        if full.exists():
            import h5py
            # bigger chunk cache: each chunk is ~64 KB, this holds ~4000 of them
            self._h5 = h5py.File(full, "r", rdcc_nbytes=256 * 1024**2, rdcc_nslots=100_003)
            self.complete = True
            self.shape = self._h5["hs"].shape
        else:
            from partial_reader import PartialMat
            part = str(full) + ".part"
            if not Path(part + ".chunks.npz").exists():
                raise FileNotFoundError(
                    f"{name}: no complete file and no chunk index; run "
                    f"src/scan_btree.py {Path(part).relative_to(ROOT)}")
            self._partial = PartialMat(part)
            self.complete = False
            self.shape = (self._partial.T, self._partial.P)

    def __repr__(self):
        src = "complete file" if self.complete else "partial download"
        return (f"Scenario({self.name!r}: {src}, {self.shape[0]:,} time steps x "
                f"{self.shape[1]:,} points, {self.available_steps():,} steps readable)")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
        if self._partial is not None:
            self._partial.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def n_points(self) -> int:
        return self.shape[1]

    def available_steps(self) -> int:
        """Time steps 0..n-1 are readable for every point and variable."""
        return self.shape[0] if self.complete else self._partial.available_steps()

    def read(self, var: str, points, steps=slice(None)) -> np.ndarray:
        """float32 array of shape (len(steps), len(points)).

        points / steps: an int, a list/array of indices (any order), or a slice.
        For many points, a slice (a contiguous range) is much faster than a list.
        """
        if var not in VARIABLES:
            raise ValueError(f"var must be one of {list(VARIABLES)}")
        if self._partial is not None:
            return self._partial.read(var, points, steps)

        assert self._h5 is not None
        d = self._h5[var]
        if isinstance(points, slice):
            cols, inv = points, None
        else:
            pts = np.atleast_1d(np.asarray(points, dtype=np.int64))
            cols, inv = np.unique(pts, return_inverse=True)   # h5py wants increasing
            cols = cols.tolist()
        if isinstance(steps, slice):
            block = d[steps, cols]
        else:
            stp = np.atleast_1d(np.asarray(steps, dtype=np.int64))
            lo, hi = int(stp.min()), int(stp.max()) + 1
            block = d[lo:hi, cols][stp - lo]
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        return block if inv is None else block[:, inv.ravel()]


def assumed_times(n: int, start: str = "1985-01-01") -> np.ndarray:
    """Timestamps for steps 0..n-1, ASSUMING 3-hourly data from `start`.

    The historical file matches 3-hourly 1985-2014 exactly, but this is not yet
    confirmed, and the ssp start dates are unknown. Label plots accordingly.
    """
    return np.datetime64(start, "h") + np.arange(n) * np.timedelta64(3, "h")


# -- grid: which location each sea point is ---------------------------------------
def load_grid(path=None, verbose: bool = True) -> dict:
    """Coordinates of the 142,868 sea points, from SeaMask_WW3.mat.

    Returns a dict with
      lon, lat          1-D grid vectors (720 and 361 values)
      mask              (lat, lon) bool array, True = sea
      point_lon/lat     coordinates of each sea point (the data's point axis)
      point_ilon/ilat   0-based grid indices of each sea point
    Assumes the data's point axis is in the same order as SeaIdx/SeaJ/SeaK;
    a map of any variable (see to_map) shows immediately if that's wrong.
    """
    import h5py
    path = Path(path) if path else RAW / "SeaMask_WW3.mat"
    with h5py.File(path, "r") as f:
        lon = np.asarray(f["lon"][()], dtype=float).ravel()
        lat = np.asarray(f["lat"][()], dtype=float).ravel()
        J = np.asarray(f["SeaJ"][()]).ravel().astype(np.int64) - 1   # MATLAB is 1-based
        K = np.asarray(f["SeaK"][()]).ravel().astype(np.int64) - 1
        idx = np.asarray(f["SeaIdx"][()]).ravel().astype(np.int64) - 1
        mask = np.asarray(f["SeaMask"][()]).astype(bool)
    n_lon, n_lat = lon.size, lat.size
    if J.max() < n_lon and K.max() < n_lat and J.max() >= n_lat:
        ilon, ilat = J, K
    elif K.max() < n_lon and J.max() < n_lat and K.max() >= n_lat:
        ilon, ilat = K, J
    else:
        raise ValueError(f"can't tell which of SeaJ/SeaK is longitude "
                         f"(max {J.max()}, {K.max()}; grid {n_lon} x {n_lat})")
    if mask.shape == (n_lon, n_lat):
        mask = mask.T
    checks = {
        "SeaIdx matches the grid indices": np.array_equal(idx, ilon + ilat * n_lon)
                                           or np.array_equal(idx, ilat + ilon * n_lat),
        "every point is sea in SeaMask": bool(mask[ilat, ilon].all()),
        "SeaMask has exactly this many sea cells": int(mask.sum()) == J.size,
    }
    if verbose:
        print(f"{J.size:,} sea points on a {n_lon} x {n_lat} grid "
              f"(lon {lon.min():g} to {lon.max():g}, lat {lat.min():g} to {lat.max():g})")
        for name, ok in checks.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    return {"lon": lon, "lat": lat, "mask": mask,
            "point_ilon": ilon, "point_ilat": ilat,
            "point_lon": lon[ilon], "point_lat": lat[ilat]}


def nearest_point(grid: dict, lat: float, lon: float) -> int:
    """Index of the sea point closest to (lat, lon). Longitude may be -180..180 or 0..360."""
    dlon = (grid["point_lon"] - lon + 180.0) % 360.0 - 180.0
    dlat = grid["point_lat"] - lat
    d2 = dlat**2 + (dlon * np.cos(np.radians(lat)))**2
    p = int(np.argmin(d2))
    print(f"point {p}: lat {grid['point_lat'][p]:g}, lon {grid['point_lon'][p]:g} "
          f"(~{111.2 * np.sqrt(d2[p]):.0f} km from the requested location)")
    return p


def to_map(grid: dict, values) -> np.ndarray:
    """One value per sea point -> (lat, lon) array for plotting; land is NaN."""
    values = np.asarray(values, dtype=float)
    if values.shape != grid["point_lon"].shape:
        raise ValueError(f"need one value per sea point ({grid['point_lon'].size:,}), "
                         f"got shape {values.shape}")
    field = np.full(grid["mask"].shape, np.nan)
    field[grid["point_ilat"], grid["point_ilon"]] = values
    return field


# -- saving results --------------------------------------------------------------
def save_fig(fig, name: str, subdir: str = "eda", formats=("png",), dpi: int = 300):
    """Save a matplotlib figure to fig/<subdir>/<name>.<ext>.

    Call before plt.show(); the figure still appears in the notebook.
    Use formats=("png", "pdf") for vector figures in the report.
    """
    out = FIG / subdir
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    print("saved:", ", ".join(str(p.relative_to(ROOT)) for p in paths))
    return paths


def save_table(df, name: str, subdir: str = "eda"):
    """Save a pandas DataFrame to data/processed/<subdir>/<name>.csv (and return it)."""
    out = PROCESSED / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.csv"
    df.to_csv(p)
    print("saved:", p.relative_to(ROOT))
    return df


# -- looking inside .mat files ---------------------------------------------------
def inspect_mat(path, max_items: int = 50) -> None:
    """Print the variables in a .mat file (v7.3/HDF5 via h5py, older via scipy)."""
    path = Path(path)
    head = path.open("rb").read(116).decode("latin-1", "replace")
    print(f"{path.name}: {head.split(',')[0].strip()}")
    if "7.3" in head:
        import h5py
        with h5py.File(path, "r") as f:
            items = []
            f.visititems(lambda n, o: items.append((n, o)))
            for n, o in items[:max_items]:
                cls = o.attrs.get("MATLAB_class", b"")
                cls = cls.decode() if isinstance(cls, bytes) else str(cls)
                if isinstance(o, h5py.Dataset):
                    mat = "x".join(map(str, reversed(o.shape))) or "scalar"
                    print(f"  {n}: MATLAB {cls or '?'} {mat}  (h5py {o.shape} {o.dtype})")
                else:
                    print(f"  {n}/  (MATLAB {cls or 'group'})")
            if len(items) > max_items:
                print(f"  ... and {len(items) - max_items} more")
    else:
        try:
            from scipy.io import whosmat
        except ImportError:
            print("  older MATLAB format: install scipy first "
                  "(.venv/bin/python -m pip install scipy)")
            return
        for name, shape, cls in whosmat(str(path)):
            print(f"  {name}: MATLAB {cls} {'x'.join(map(str, shape))}")
