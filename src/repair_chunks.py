"""Re-download the damaged index nodes and chunks found by scan_chunks.py and patch them in.

Principle
    A damaged chunk is a known byte range in the file. The provider's link answers HTTP
    range requests, so only those bytes are fetched again (kilobytes, not 60 GB). The
    fetched bytes are accepted only if every chunk in them now decompresses and every
    index node in them is a valid node; the old
    bytes are backed up first, the new bytes written in place, then read back and
    checked again. The file size never changes.

Ranges that are already sound on disk are skipped, so the script can simply be rerun
    (or looped) after a connection failure; only what is still damaged is fetched.

Usage (from the project root; uses the links in data/raw/urls.env):
    .venv/bin/python src/repair_chunks.py ssp126 --dry-run   # fetch + verify, write nothing
    .venv/bin/python src/repair_chunks.py ssp126             # fetch + verify + patch
Afterwards: rerun scan_chunks.py on that scenario (expect 0 damaged) and redo its sha256.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

import requests

from scan_chunks import parse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL_VAR = {"historical": "URL_HIST", "ssp126": "URL_126", "ssp585": "URL_585"}
EXPECTED_RAW = 248 * 66 * 4          # bytes of one decompressed chunk
MERGE_GAP = 1 << 20                  # fetch neighbouring damaged chunks in one request
RETRIES = 12


def hide(text) -> str:
    return re.sub(r"access_token=[^&\s'\"]*", "access_token=***", str(text))


def link(scenario: str) -> str:
    env = RAW / "urls.env"
    var = URL_VAR[scenario]
    url = subprocess.run(["bash", "-c", f'source "{env}" && printf %s "${var}"'],
                         capture_output=True, text=True, check=True).stdout.strip()
    if not url.startswith("http"):
        sys.exit(f"{var} not found in {env}")
    m = re.search(r"access_token=[^&]*?(\d+)(?:&|$)", url)
    if m and int(m[1]) < time.time():
        sys.exit(f"{var} expired at {time.ctime(int(m[1]))}: put a fresh link in {env}")
    return url


def fetch(session, url, a, b):
    """Bytes a..b-1 of the remote file, insisting on a correct 206 answer."""
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
            cr = r.headers.get("Content-Range", "")
            if r.status_code != 206 or not cr.startswith(f"bytes {a}-"):
                raise IOError(f"HTTP {r.status_code}, Content-Range '{cr}', "
                              f"type {r.headers.get('Content-Type')}")
            if len(r.content) != b - a:
                raise IOError(f"got {len(r.content)} bytes, expected {b - a}")
            return r.content
        except (requests.RequestException, IOError) as e:
            wait = min(2 ** attempt, 120)
            print(f"    attempt {attempt}: {hide(e)[:150]} - retry in {wait}s", flush=True)
            time.sleep(wait)
    sys.exit("Giving up on this range; nothing was written. Re-run later (or with a fresh link).")


def chunks_ok(buf, base, items, file_size):
    """True if every listed item inside buf (which starts at file offset base) is sound:
    chunks must decompress, index nodes must parse as valid nodes."""
    for off, size, kind in items:
        piece = buf[off - base: off - base + size]
        if kind == "node":
            if parse(piece, 0, len(piece), file_size) is None:
                return False
            continue
        try:
            if len(zlib.decompress(piece)) != EXPECTED_RAW:
                return False
        except zlib.error:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Re-download and patch damaged chunks.")
    ap.add_argument("scenario", choices=list(URL_VAR))
    ap.add_argument("--dry-run", action="store_true", help="fetch and verify, but write nothing")
    a = ap.parse_args()

    path = RAW / a.scenario / "EC-EARTH3.mat"
    rows = list(csv.DictReader(open(RAW / a.scenario / "bad_chunks.csv")))
    if not rows:
        print("bad_chunks.csv lists no damaged chunks: nothing to do.")
        return
    bad = sorted((int(r["byte_offset"]), int(r["nbytes"]), r.get("kind", "chunk")) for r in rows)
    file_size = path.stat().st_size

    # merge nearby damaged chunks into a few ranges
    ranges = []
    for off, size, kind in bad:
        size = min(size, file_size - off)
        if ranges and off - ranges[-1][1] <= MERGE_GAP:
            ranges[-1][1] = max(ranges[-1][1], off + size)
            ranges[-1][2].append((off, size, kind))
        else:
            ranges.append([off, off + size, [(off, size, kind)]])
    print(f"{a.scenario}: {len(bad)} damaged item(s) "
          f"({sum(k == 'node' for *_, k in bad)} index nodes) in {len(ranges)} range(s), "
          f"{sum(b - s for s, b, _ in ranges):,} bytes to fetch", flush=True)

    url = link(a.scenario)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (python-requests downloader)"
    backup = RAW / a.scenario / "repair_backup"
    fd = os.open(path, os.O_RDONLY if a.dry_run else os.O_RDWR)
    try:
        for s, e, chunks in ranges:
            print(f"  bytes {s:,}-{e - 1:,} ({len(chunks)} item(s))", flush=True)
            if chunks_ok(os.pread(fd, e - s, s), s, chunks, file_size):
                print("    already sound on disk (repaired earlier) - skipping", flush=True)
                continue
            new = fetch(session, url, s, e)
            if not chunks_ok(new, s, chunks, file_size):
                sys.exit("    downloaded bytes are ALSO damaged: the provider's copy may be bad. "
                         "Nothing was written - tell the data provider.")
            old = os.pread(fd, e - s, s)
            if old == new:
                print("    local bytes already equal the remote bytes - nothing to fix here", flush=True)
                continue
            print(f"    verified: all items are sound; {sum(x != y for x, y in zip(old, new)):,} "
                  f"bytes differ from the local copy", flush=True)
            if a.dry_run:
                continue
            backup.mkdir(exist_ok=True)
            (backup / f"{s}.bin").write_bytes(old)
            os.pwrite(fd, new, s)
            os.fsync(fd)
            if os.pread(fd, e - s, s) != new or not chunks_ok(os.pread(fd, e - s, s), s, chunks, file_size):
                sys.exit("    read-back check FAILED - restore from repair_backup/ and investigate")
            print("    written and read back OK", flush=True)
    finally:
        os.close(fd)
    print("dry run finished, nothing written" if a.dry_run else
          "done: rerun scan_chunks.py on this scenario, then sha256sum", flush=True)


if __name__ == "__main__":
    main()
