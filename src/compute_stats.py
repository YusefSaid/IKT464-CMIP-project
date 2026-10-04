"""Full-period statistics at every sea point of the EC-Earth3 wave files.

Objective
    Reduce each 30-year, 3-hourly file (one variable is ~47 GiB in memory) to small
    per-point statistics that the notebooks load in seconds.

Principle
    The data are stored in chunks of 248 time steps x 66 sea points. Reading all time
    steps for a block of neighbouring points (a multiple of 66) touches each chunk once,
    so the whole file is read exactly once and only one block is in memory per worker.
    Month boundaries come from the file's own metadata (FileNames, MonthStartIndex,
    MonthEndIndex), so yearly and monthly aggregates follow the true calendar.
    Every chunk is decompressed, so a corrupted chunk raises an error here: a complete
    run is also a full integrity check of the file.

Output per scenario and variable: data/processed/stats30/<scenario>_<var>.npz with
    per point (P,):      mean, std, p50, p90, p99, nan_frac
    per year (Y, P):     annual_mean, annual_max, annual_valid (fraction of valid steps)
    per month (12, P):   monthly_mean (climatology, Jan..Dec)
    axes:                years, months_of_year, quantiles

Run from the project root, in the background (one log per run):
    mkdir -p data/processed/stats30
    nohup .venv/bin/python src/compute_stats.py --vars hs \
        > data/processed/stats30/run_hs.log 2>&1 &
    tail -f data/processed/stats30/run_hs.log

Existing outputs are skipped (use --force to recompute), so a restarted workspace
only loses the scenario/variable that was running.
"""
from __future__ import annotations

import argparse
import calendar
import os
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "stats30"
SCENARIOS = ("historical", "ssp126", "ssp585")
QUANTILES = (50, 90, 99)
CHUNK_POINTS = 66        # sea points per storage chunk
STEPS_PER_DAY = 8        # 3-hourly


def _tp(fp):
    """Peak period Tp = 1/fp [s]; NaN where fp is missing or not positive."""
    out = np.full(fp.shape, np.nan, np.float32)
    np.divide(1.0, fp, out=out, where=fp > 0)
    return out


# variable name -> (stored variables needed, function computing it)
VARIABLES = {
    "hs":   (("hs",), lambda hs: hs),                          # significant wave height [m]
    "fp":   (("fp",), lambda fp: fp),                          # peak frequency [Hz]
    "tp":   (("fp",), _tp),                                    # peak period [s]
    "uwnd": (("uwnd",), lambda u: u),                          # eastward wind [m/s]
    "vwnd": (("vwnd",), lambda v: v),                          # northward wind [m/s]
    "wspd": (("uwnd", "vwnd"), lambda u, v: np.hypot(u, v)),   # wind speed [m/s]
}


def raw_path(scenario: str) -> Path:
    return RAW / scenario / "EC-EARTH3.mat"


# --------------------------------------------------------------------------- time axis
def _matlab_char(ds) -> str:
    return "".join(map(chr, np.ravel(ds[()]))).strip("'")


def time_axis(path) -> dict:
    """Calendar of a file, read from its MATLAB metadata struct.

    Returns year and month of each of the stored months, their 0-based start step and
    end step (exclusive), and checks that the months are consecutive, start in January,
    cover all time steps, and have 8 steps per calendar day (leap days included).
    """
    with h5py.File(path, "r") as f:
        m = f["metadata"]
        names = [_matlab_char(f[r]) for r in np.ravel(m["FileNames"][()])]
        start = np.ravel(m["MonthStartIndex"][()]).astype(np.int64) - 1   # MATLAB is 1-based
        end = np.ravel(m["MonthEndIndex"][()]).astype(np.int64)           # inclusive -> exclusive
        n_steps = int(np.ravel(m["Ntime"][()])[0])
        scenario = _matlab_char(m["ScenarioName"])
    year = np.array([int(n[:4]) for n in names])
    month = np.array([int(n[4:6]) for n in names])

    if not (start[0] == 0 and end[-1] == n_steps and np.array_equal(start[1:], end[:-1])):
        raise ValueError(f"{path}: month index does not cover the time steps contiguously")
    if not np.all(np.diff(year * 12 + month) == 1):
        raise ValueError(f"{path}: months are not consecutive")
    if month[0] != 1 or len(month) % 12:
        raise ValueError(f"{path}: expected whole years starting in January")
    days = np.array([calendar.monthrange(y, mo)[1] for y, mo in zip(year, month)])
    gregorian = bool(np.array_equal(end - start, days * STEPS_PER_DAY))

    return {"scenario": scenario, "year": year, "month": month, "start": start, "end": end,
            "n_steps": n_steps, "years": np.unique(year), "gregorian_calendar": gregorian,
            "leap_days": int(np.sum((month == 2) & (end - start == 29 * STEPS_PER_DAY)))}


# --------------------------------------------------------------------------- statistics
def block_stats(x: np.ndarray, month_start: np.ndarray) -> dict:
    """Statistics of one block x (time, points); month_start = first step of each month."""
    n_months, n_pts = len(month_start), x.shape[1]
    n_years = n_months // 12
    valid = ~np.isnan(x)
    xz = np.where(valid, x, 0.0).astype(np.float32, copy=False)

    # monthly sums -> everything time-aggregated (float64 sums for accuracy)
    cnt = np.add.reduceat(valid, month_start, axis=0, dtype=np.int64)          # (M, B)
    s1 = np.add.reduceat(xz, month_start, axis=0, dtype=np.float64)
    s2 = np.add.reduceat(xz * xz, month_start, axis=0, dtype=np.float64)
    mx = np.fmax.reduceat(x, month_start, axis=0)                              # NaN-ignoring max
    del xz

    with np.errstate(invalid="ignore", divide="ignore"):
        n = cnt.sum(0)
        mean = s1.sum(0) / n
        std = np.sqrt(np.maximum(s2.sum(0) / n - mean ** 2, 0.0))
        cy = cnt.reshape(n_years, 12, n_pts)
        annual_mean = s1.reshape(n_years, 12, n_pts).sum(1) / cy.sum(1)
        monthly_mean = s1.reshape(n_years, 12, n_pts).sum(0) / cy.sum(0)
    annual_max = np.fmax.reduce(mx.reshape(n_years, 12, n_pts), axis=1)
    steps_per_year = np.add.reduceat(np.ones(x.shape[0], np.int64),
                                     month_start, dtype=np.int64).reshape(n_years, 12).sum(1)
    annual_valid = cy.sum(1) / steps_per_year[:, None]

    # percentiles: fast path for points without gaps, nan-aware path otherwise
    q = np.full((len(QUANTILES), n_pts), np.nan)
    full = valid.all(0)
    part = ~full & valid.any(0)
    if full.any():
        q[:, full] = np.percentile(x[:, full], QUANTILES, axis=0)
    if part.any():
        q[:, part] = np.nanpercentile(x[:, part], QUANTILES, axis=0)

    out = {"mean": mean, "std": std, "nan_frac": 1.0 - n / x.shape[0],
           "annual_mean": annual_mean, "annual_max": annual_max,
           "annual_valid": annual_valid, "monthly_mean": monthly_mean}
    out.update({f"p{qq}": q[i] for i, qq in enumerate(QUANTILES)})
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


# --------------------------------------------------------------------------- workers
_FILE = None
_MONTH_START = None


def _init_worker(path: str, month_start: np.ndarray):
    global _FILE, _MONTH_START
    _FILE = h5py.File(path, "r")
    _MONTH_START = month_start


def _work(task):
    var, b0, b1 = task
    stored, fn = VARIABLES[var]
    try:
        x = fn(*(_FILE[v][:, b0:b1] for v in stored)).astype(np.float32, copy=False)
    except Exception as e:                      # e.g. a corrupted chunk
        raise RuntimeError(f"reading {var} at sea points {b0}-{b1 - 1} failed: {e!r}") from e
    return b0, b1, block_stats(x, _MONTH_START)


def run(scenario: str, var: str, workers: int, block: int, force: bool = False) -> Path:
    out = OUT / f"{scenario}_{var}.npz"
    if out.exists() and not force:
        print(f"{out.name} exists, skipping (use --force to recompute)", flush=True)
        return out
    path = raw_path(scenario)
    tax = time_axis(path)
    with h5py.File(path, "r") as f:
        n_steps, n_pts = f[VARIABLES[var][0][0]].shape
    if n_steps != tax["n_steps"]:
        raise ValueError(f"{path}: {n_steps} steps in data but {tax['n_steps']} in metadata")

    n_years = len(tax["years"])
    shapes = {"annual_mean": (n_years, n_pts), "annual_max": (n_years, n_pts),
              "annual_valid": (n_years, n_pts), "monthly_mean": (12, n_pts)}
    res = {}
    tasks = [(var, b0, min(b0 + block, n_pts)) for b0 in range(0, n_pts, block)]
    print(f"{scenario} {var}: {tax['years'][0]}-{tax['years'][-1]}, {n_steps:,} steps x "
          f"{n_pts:,} points, {len(tasks)} blocks, {workers} workers", flush=True)

    t0, done, next_report = time.time(), 0, 0.05
    with Pool(workers, initializer=_init_worker, initargs=(str(path), tax["start"])) as pool:
        for b0, b1, r in pool.imap_unordered(_work, tasks):
            for k, v in r.items():
                if k not in res:
                    res[k] = np.full(shapes.get(k, (n_pts,)), np.nan, np.float32)
                res[k][..., b0:b1] = v
            done += b1 - b0
            frac = done / n_pts
            if frac >= next_report or done == n_pts:
                el = time.time() - t0
                print(f"  {100 * frac:5.1f}%  {el / 60:6.1f} min elapsed, "
                      f"~{el / frac * (1 - frac) / 60:6.1f} min left", flush=True)
                next_report += 0.05

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res, years=tax["years"], months_of_year=np.arange(1, 13),
                        quantiles=np.array(QUANTILES), scenario=scenario, var=var)
    os.replace(tmp, out)                        # never leave a half-written result
    print(f"saved {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return out


def load_stats(scenario: str, var: str = "hs") -> dict:
    """Load the statistics written by run() as a dict of arrays."""
    with np.load(OUT / f"{scenario}_{var}.npz") as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument("--vars", nargs="+", default=["hs"], choices=list(VARIABLES))
    ap.add_argument("--workers", type=int, default=5, help="processes (6 cores available)")
    ap.add_argument("--block", type=int, default=5 * CHUNK_POINTS,
                    help="sea points per task, a multiple of 66 (default 330)")
    ap.add_argument("--force", action="store_true", help="recompute existing outputs")
    a = ap.parse_args()
    if a.block % CHUNK_POINTS:
        ap.error(f"--block must be a multiple of {CHUNK_POINTS}")
    for var in a.vars:
        for s in a.scenarios:
            run(s, var, a.workers, a.block, a.force)


if __name__ == "__main__":
    main()
