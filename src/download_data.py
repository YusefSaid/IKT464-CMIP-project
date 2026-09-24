#!/usr/bin/env python3
"""
Resumable, chunked downloader for large files (e.g. a 56 GB .mat export
served from a myQNAPcloud share link).

Downloads to <dest>.part and only renames to <dest> once the byte count
matches Content-Length, so a half-finished file can never be mistaken
for a complete one. Re-running the script resumes where it left off.

Usage
-----
    python download_data.py --url "<direct download url>" \
                            --dest data/raw/EC-EARTH3.mat

    # or put the url in a file / env var so it stays out of your shell history
    export QNAP_URL="https://...."
    python download_data.py --dest data/raw/EC-EARTH3.mat

Long downloads should be run detached:
    nohup python download_data.py --dest data/raw/EC-EARTH3.mat > dl.log 2>&1 &
    tail -f dl.log

Parallel connections (for servers that throttle each connection):
    python download_data.py --url "$URL" --dest data/raw/EC-EARTH3.mat \
                            --connections 4
    Any existing single-stream .part is kept as the first segment. Re-running
    the same command resumes every segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except ImportError:  # pip install tqdm -- falls back to plain log lines
    tqdm = None

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per write
MAX_RETRIES = 100             # per stalled connection, not total
BACKOFF_CAP = 60.0            # seconds
TIMEOUT = (30, 120)           # (connect, read)
SEG_CHUNK = 1024 * 1024       # 1 MiB writes in segmented mode, for smoother progress
MIN_SEGMENT = int(os.environ.get("DL_MIN_SEGMENT", 64 * 1024 * 1024))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def parse_curl(text: str) -> tuple[str | None, dict[str, str]]:
    """Pull the URL and headers out of a DevTools 'Copy as cURL (bash)' string.

    Chrome puts cookies in -H 'Cookie: ...' or -b, which is exactly the
    session state a QNAP share link needs. Headers that would fight with
    our own resume logic are dropped.
    """
    import shlex

    cleaned = text.replace("\\\n", " ").replace("^\n", " ").replace("`\n", " ")
    try:
        tokens = shlex.split(cleaned)
    except ValueError as exc:
        sys.exit(f"Could not parse the curl command ({exc}). "
                 "Use 'Copy as cURL (bash)', not the cmd/PowerShell variant.")

    url: str | None = None
    headers: dict[str, str] = {}
    skip_with_arg = {"-X", "--request", "-d", "--data", "--data-raw",
                     "--data-binary", "--data-urlencode", "-o", "--output",
                     "--max-time", "--connect-timeout", "--retry"}

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "curl":
            i += 1
        elif t in ("-H", "--header") and i + 1 < len(tokens):
            key, _, val = tokens[i + 1].partition(":")
            headers[key.strip()] = val.strip()
            i += 2
        elif t in ("-b", "--cookie") and i + 1 < len(tokens):
            headers["Cookie"] = tokens[i + 1]
            i += 2
        elif t in ("-A", "--user-agent") and i + 1 < len(tokens):
            headers["User-Agent"] = tokens[i + 1]
            i += 2
        elif t in ("-e", "--referer") and i + 1 < len(tokens):
            headers["Referer"] = tokens[i + 1]
            i += 2
        elif t in skip_with_arg:
            i += 2
        elif t.startswith("-"):
            i += 1
        elif t.lower().startswith(("http://", "https://")) and url is None:
            url = t
            i += 1
        else:
            i += 1

    # These would break Range requests or make Content-Length meaningless.
    for bad in ("range", "if-range", "accept-encoding", "content-length", "host"):
        for key in [k for k in headers if k.lower() == bad]:
            headers.pop(key)

    return url, headers


def is_fatal(exc: BaseException) -> str | None:
    """Return a message if the error will never fix itself, else None.

    Retrying a typo'd URL or a 404 just wastes minutes — only connection
    and server-side hiccups are worth backing off and trying again.
    """
    if isinstance(exc, (requests.exceptions.MissingSchema,
                        requests.exceptions.InvalidSchema,
                        requests.exceptions.InvalidURL,
                        requests.exceptions.URLRequired)):
        return f"Bad URL: {exc}"
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
        if code and 400 <= code < 500 and code not in (408, 429):
            hint = {
                401: "the share needs authentication",
                403: "the link has expired, or needs a password/cookie",
                404: "the file or share id no longer exists",
            }.get(code, "the server rejected the request")
            return f"HTTP {code} — {hint}."
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
        return "Out of disk space."
    return None


class Progress:
    """A tqdm bar when stderr is a terminal, periodic log lines when it isn't.

    A live bar redrawn with carriage returns turns a nohup log into an
    unreadable mess, so 'auto' picks lines whenever output is redirected.
    """

    def __init__(self, total: int | None, initial: int = 0, style: str = "auto",
                 desc: str = "", interval: float = 5.0):
        if style == "auto":
            style = "bar" if (tqdm is not None and sys.stderr.isatty()) else "lines"
        if style == "bar" and tqdm is None:
            print("tqdm not installed — falling back to log lines "
                  "(pip install tqdm).", flush=True)
            style = "lines"

        self.style = style
        self.total = total
        self.n = initial
        self._n0 = initial
        self._t0 = time.monotonic()
        self._last = 0.0
        self._interval = interval
        self._bar = None

        if style == "bar" and tqdm is not None:
            self._bar = tqdm(
                total=total, initial=initial, desc=desc or None,
                unit="B", unit_scale=True, unit_divisor=1024,
                smoothing=0.05, miniters=1,
                bar_format="{desc}{percentage:5.1f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )

    def update(self, nbytes: int) -> None:
        self.n += nbytes
        if self._bar is not None:
            self._bar.update(nbytes)
            return
        if self.style == "none":
            return
        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        rate = (self.n - self._n0) / max(now - self._t0, 1e-6)
        if self.total:
            eta = (self.total - self.n) / rate if rate else 0.0
            print(f"  {human(self.n)} / {human(self.total)}  "
                  f"({100.0 * self.n / self.total:5.1f}%)  "
                  f"{human(rate)}/s  ETA {eta / 3600:.1f} h", flush=True)
        else:
            print(f"  {human(self.n)}  {human(rate)}/s", flush=True)

    def rewind(self, value: int = 0) -> None:
        """Jump the counter back, e.g. when a server forces a restart from 0."""
        self.n = value
        self._n0 = value
        self._t0 = time.monotonic()
        if self._bar is not None:
            self._bar.reset(total=self.total)
            self._bar.update(value)

    def set_total(self, total: int | None) -> None:
        self.total = total
        if self._bar is not None and total:
            self._bar.total = total
            self._bar.refresh()

    def write(self, msg: str) -> None:
        """Print without tearing the bar apart."""
        if self._bar is not None:
            self._bar.write(msg)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def content_type(session: requests.Session, url: str) -> str | None:
    """Peek at the first byte to see what the server actually serves."""
    try:
        r = session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                        allow_redirects=True, timeout=TIMEOUT)
        ct = r.headers.get("Content-Type")
        r.close()
        return ct
    except requests.RequestException as exc:
        fatal = is_fatal(exc)
        if fatal:
            sys.exit(fatal)
        return None


def remote_size(session: requests.Session, url: str) -> int | None:
    """Content-Length of the full file, or None if the server won't say."""
    for method in ("HEAD", "GET"):
        try:
            r = session.request(
                method, url, allow_redirects=True, timeout=TIMEOUT,
                stream=(method == "GET"),
                headers={"Range": "bytes=0-0"} if method == "GET" else None,
            )
            if method == "GET":
                # 206 -> Content-Range: bytes 0-0/12345
                cr = r.headers.get("Content-Range", "")
                r.close()
                if "/" in cr:
                    total = cr.rsplit("/", 1)[1]
                    if total.isdigit():
                        return int(total)
            else:
                r.close()
                if r.status_code != 200:   # an error page's length is not the file's
                    continue
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
        except requests.RequestException:
            continue
    return None


def check_disk_space(dest: Path, needed: int | None) -> None:
    if needed is None:
        return
    free = os.statvfs(dest.parent).f_frsize * os.statvfs(dest.parent).f_bavail
    already = dest.with_suffix(dest.suffix + ".part").stat().st_size \
        if dest.with_suffix(dest.suffix + ".part").exists() else 0
    required = needed - already
    if free < required * 1.02:
        sys.exit(
            f"Not enough disk space on {dest.parent}: "
            f"need ~{human(required)}, have {human(free)}"
        )


class BadResponse(Exception):
    """The server answered, but not with the file bytes that were asked for."""


def hide_tokens(text: str) -> str:
    """Mask access tokens so they never end up in logs."""
    return re.sub(r"access_token=[^&\s'\"]*", "access_token=***", str(text))


def parse_content_range(value: str) -> tuple[int, int, int] | None:
    """'bytes 100-199/1000' -> (100, 199, 1000); anything else -> None."""
    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def download(url: str, dest: Path, expect_md5: str | None = None,
             progress: str = "auto", extra_headers: dict[str, str] | None = None,
             expect_size: int | None = None) -> None:
    """Single-connection download that can never discard data it already has.

    Every request asks for a byte range and must be answered with HTTP 206 and
    a Content-Range starting exactly at the byte we asked for. Anything else --
    an error page, a 200 that ignores the range, a text/HTML body, a range at
    the wrong offset -- is treated as a temporary failure and retried, and the
    partial file is left untouched. The .part file is renamed to its final
    name only when its size equals the size the server (or --size) states.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (python-requests downloader)"
    if extra_headers:
        session.headers.update(extra_headers)
        print(f"Using {len(extra_headers)} captured header(s): "
              f"{', '.join(sorted(extra_headers))}", flush=True)

    total = expect_size
    if dest.exists():
        size = dest.stat().st_size
        if total is None:
            sys.exit(f"{dest} already exists ({human(size)}). Pass --size to check "
                     f"it against the expected size, or delete it to download again.")
        if size == total:
            print(f"{dest} is already complete.", flush=True)
            return
        sys.exit(f"{dest} exists but is {size:,} bytes, not the expected {total:,}. "
                 f"It is NOT a complete download: delete it and re-run.")

    have = part.stat().st_size if part.exists() else 0
    if total is not None and have > total:
        sys.exit(f"{part} ({have:,} bytes) is larger than the expected {total:,} bytes. "
                 f"Nothing was changed; check that --size and the link are right.")
    print(f"Expected size: {human(total) if total else 'from the server'}; "
          f"already have {human(have)}", flush=True)
    check_disk_space(dest, total)

    attempt = 0
    t_start = time.monotonic()
    bar = Progress(total, initial=have, style=progress, desc=dest.name + " ")

    while True:
        done = part.stat().st_size if part.exists() else 0
        if total is not None and done == total:
            break

        try:
            r = session.get(url, headers={"Range": f"bytes={done}-"}, stream=True,
                            allow_redirects=True, timeout=TIMEOUT)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
            cr = parse_content_range(r.headers.get("Content-Range", ""))

            if r.status_code != 206 or cr is None:
                r.close()
                raise BadResponse(f"expected HTTP 206 with a Content-Range, got HTTP "
                                  f"{r.status_code} ({ctype or 'no content type'})")
            if ctype.startswith(("text/", "application/json", "application/xhtml")):
                r.close()
                raise BadResponse(f"server sent {ctype} instead of file data")
            start, _end, remote_total = cr
            if start != done:
                r.close()
                raise BadResponse(f"server resumed at byte {start:,}, not {done:,}")

            if total is None:
                total = remote_total
                bar.set_total(total)
                bar.write(f"Remote size: {human(total)} ({total:,} bytes)")
            elif remote_total != total:
                r.close()
                bar.close()
                sys.exit(f"The server reports a size of {remote_total:,} bytes, but "
                         f"{total:,} was expected. Nothing was deleted -- check the "
                         f"link points at the right file.")

            bar.rewind(done)
            with open(part, "ab") as fh:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if not chunk:
                        continue
                    chunk = chunk[: total - done]   # never write past the end
                    fh.write(chunk)
                    done += len(chunk)
                    bar.update(len(chunk))
                    if done >= total:
                        break
            r.close()
            attempt = 0  # a successful exchange resets the retry budget

        except (requests.RequestException, OSError, BadResponse) as exc:
            fatal = None if isinstance(exc, BadResponse) else is_fatal(exc)
            if fatal:
                bar.close()
                sys.exit(hide_tokens(fatal) + "\nThe partial file is kept; re-run to "
                         "resume (with a fresh link if it expired).")
            attempt += 1
            if attempt > MAX_RETRIES:
                bar.close()
                sys.exit(f"Giving up after {MAX_RETRIES} retries: {hide_tokens(exc)}\n"
                         f"The partial file is kept; re-run to resume.")
            wait = min(2 ** min(attempt, 6), BACKOFF_CAP)
            bar.write(f"[retry {attempt}] {type(exc).__name__}: "
                      f"{hide_tokens(exc)[:200]} -- resuming in {wait:.0f}s")
            time.sleep(wait)

    bar.close()
    final = part.stat().st_size
    if final != total:
        sys.exit(f"Size mismatch: have {final:,} bytes, expected {total:,}. "
                 f"The partial file is kept.")

    if expect_md5:
        print("Verifying md5 (this reads the whole file)...", flush=True)
        h = hashlib.md5()
        vbar = Progress(final, style=progress, desc="md5 ")
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK_SIZE), b""):
                h.update(block)
                vbar.update(len(block))
        vbar.close()
        if h.hexdigest() != expect_md5.lower():
            sys.exit(f"md5 mismatch: {h.hexdigest()} != {expect_md5}")
        print("md5 ok", flush=True)

    part.rename(dest)
    elapsed = time.monotonic() - t_start
    print(f"Done: {dest} ({human(final)}, {final:,} bytes) in {elapsed/3600:.2f} h",
          flush=True)


def probe(session: requests.Session, url: str) -> tuple[int, str, int | None]:
    """One ranged GET: (status, content-type, total size from Content-Range)."""
    r = session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                    allow_redirects=True, timeout=TIMEOUT)
    r.raise_for_status()
    status = r.status_code
    ctype = r.headers.get("Content-Type", "")
    cr = r.headers.get("Content-Range", "")
    r.close()
    tail = cr.rsplit("/", 1)[-1] if "/" in cr else ""
    return status, ctype, (int(tail) if tail.isdigit() else None)


def segment_paths(dest: Path) -> tuple[Path, Path]:
    return (dest.with_suffix(dest.suffix + ".part"),
            dest.with_suffix(dest.suffix + ".segments.json"))


def download_segmented(url: str, dest: Path, connections: int,
                       expect_md5: str | None = None, progress: str = "auto",
                       extra_headers: dict[str, str] | None = None) -> None:
    """Fetch one file over several parallel ranged connections.

    The relay throttles per connection, so N connections to different parts
    of the same file multiply throughput. Layout on disk:

        <dest>.part            segment 0 (also any earlier single-stream progress)
        <dest>.part1 .. partN  the remaining segments
        <dest>.segments.json   the segment boundaries, so a re-run resumes
                               every segment exactly where it stopped

    When all segments are complete they are appended onto <dest>.part in
    order and the result is renamed to <dest>. The merge is restartable:
    each step truncates to the known boundary before appending, and its
    progress is recorded before the segment file is deleted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, meta_path = segment_paths(dest)

    def new_session() -> requests.Session:
        s = requests.Session()
        s.headers["User-Agent"] = "Mozilla/5.0 (python-requests downloader)"
        if extra_headers:
            s.headers.update(extra_headers)
        return s

    try:
        status, ctype, total = probe(new_session(), url)
    except requests.RequestException as exc:
        sys.exit(is_fatal(exc) or f"Could not reach the server: {exc}")

    if ctype.split(";")[0].strip() in ("text/html", "application/xhtml+xml"):
        sys.exit("The server returned an HTML page, not a file — check the link.")
    if status != 206 or not total:
        sys.exit("The server does not support range requests for this link, so a "
                 "segmented download is impossible. Run without --connections.")
    print(f"Remote size: {human(total)}", flush=True)

    if dest.exists() and dest.stat().st_size == total:
        print(f"{dest} already complete.", flush=True)
        return

    # ---- segment plan: reuse the saved one, or make a new one -------------
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("total") != total:
            sys.exit(f"The remote file changed size since this download began "
                     f"({meta.get('total')} -> {total}). Delete {part.name}*, "
                     f"{meta_path.name} and start again.")
        bounds = [tuple(b) for b in meta["bounds"]]
        if connections != len(bounds):
            print(f"Resuming the existing {len(bounds)}-segment plan "
                  f"(--connections {connections} ignored).", flush=True)
    else:
        have0 = part.stat().st_size if part.exists() else 0
        if have0 > total:
            sys.exit(f"{part} is larger than the remote file — delete it and restart.")
        rest = total - have0
        n = max(1, min(connections, rest // MIN_SEGMENT or 1))
        step = rest // n
        edges = [have0 + i * step for i in range(n)] + [total]
        bounds = [(0 if i == 0 else edges[i], edges[i + 1]) for i in range(n)]
        meta = {"total": total, "bounds": bounds, "merging": False, "merged": 0}
        meta_path.write_text(json.dumps(meta))
        if have0:
            print(f"Keeping {human(have0)} already downloaded as the start of "
                  f"segment 1.", flush=True)

    files = [part] + [dest.with_suffix(dest.suffix + f".part{i}")
                      for i in range(1, len(bounds))]

    def have(i: int) -> int:
        return files[i].stat().st_size if files[i].exists() else 0

    # ---- download phase ---------------------------------------------------
    t_start = time.monotonic()
    if not meta.get("merging"):
        remaining = sum((e - s) - have(i) for i, (s, e) in enumerate(bounds))
        free = shutil.disk_usage(dest.parent).free
        if free < remaining * 1.02:
            sys.exit(f"Not enough disk space: need ~{human(remaining)}, "
                     f"have {human(free)}")

        print(f"Downloading over {len(bounds)} connection(s):", flush=True)
        for i, (s, e) in enumerate(bounds):
            print(f"  segment {i + 1}: {human(e - s):>9}  "
                  f"({100 * have(i) / (e - s):5.1f}% done)", flush=True)

        lock = threading.Lock()
        stop = threading.Event()
        errors: list[str] = []
        bar = Progress(total, initial=total - remaining, style=progress,
                       desc=dest.name + " ")

        def fail(msg: str) -> None:
            with lock:
                errors.append(msg)
            stop.set()

        def worker(i: int) -> None:
            s, e = bounds[i]
            sess = new_session()
            attempt = 0
            while not stop.is_set():
                pos = s + have(i)
                if pos == e:
                    return
                if pos > e:
                    return fail(f"segment {i + 1} is larger than planned — "
                                f"delete {files[i].name} and re-run.")
                try:
                    r = sess.get(url, headers={"Range": f"bytes={pos}-{e - 1}"},
                                 stream=True, allow_redirects=True, timeout=TIMEOUT)
                    r.raise_for_status()
                    if r.status_code != 206:
                        r.close()
                        return fail("The server stopped honouring range requests.")
                    with open(files[i], "ab") as fh:
                        for chunk in r.iter_content(SEG_CHUNK):
                            if stop.is_set():
                                break
                            if not chunk:
                                continue
                            chunk = chunk[: e - pos]   # never write past the boundary
                            fh.write(chunk)
                            pos += len(chunk)
                            with lock:
                                bar.update(len(chunk))
                            if pos >= e:
                                break
                    r.close()
                    attempt = 0
                except (requests.RequestException, OSError) as exc:
                    fatal = is_fatal(exc)
                    if fatal:
                        return fail(fatal)
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        return fail(f"segment {i + 1} gave up after "
                                    f"{MAX_RETRIES} retries: {exc}")
                    wait = min(2 ** min(attempt, 6), BACKOFF_CAP)
                    with lock:
                        bar.write(f"[segment {i + 1} retry {attempt}] "
                                  f"{type(exc).__name__} — resuming in {wait:.0f}s")
                    stop.wait(wait)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(len(bounds))]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            stop.set()
            bar.close()
            sys.exit("Interrupted — re-run the same command to resume.")
        bar.close()

        if errors:
            sys.exit(errors[0] + "\nProgress is saved; re-run the same command "
                                 "to resume (with a fresh link if it expired).")

        for i, (s, e) in enumerate(bounds):
            if have(i) != e - s:
                sys.exit(f"segment {i + 1} is incomplete ({have(i)} of {e - s} "
                         f"bytes). Re-run to resume.")

        meta["merging"] = True
        meta_path.write_text(json.dumps(meta))

    # ---- merge phase (restartable) ----------------------------------------
    if len(bounds) > 1:
        print("All segments complete — joining them "
              "(a few minutes of disk I/O)...", flush=True)
    for i in range(1, len(bounds)):
        if i <= meta.get("merged", 0):
            files[i].unlink(missing_ok=True)   # merged earlier; drop leftovers
            continue
        s, e = bounds[i]
        seg = files[i]
        if not seg.exists() or seg.stat().st_size != e - s:
            sys.exit(f"{seg.name} is missing or the wrong size; cannot join. "
                     f"Delete {meta_path.name} and the .part files to start over.")
        if part.stat().st_size < s:
            sys.exit(f"{part.name} is shorter than expected before joining "
                     f"segment {i + 1}.")
        with open(part, "r+b") as out:
            out.truncate(s)          # idempotent if a previous join was cut short
            out.seek(s)
            with open(seg, "rb") as src:
                shutil.copyfileobj(src, out, CHUNK_SIZE)
            out.flush()
            os.fsync(out.fileno())
        meta["merged"] = i
        meta_path.write_text(json.dumps(meta))
        seg.unlink()

    final = part.stat().st_size
    if final != total:
        sys.exit(f"Size mismatch after joining: {final} vs {total}. Files kept.")

    if expect_md5:
        print("Verifying md5 (this reads the whole file)...", flush=True)
        h = hashlib.md5()
        with open(part, "rb") as fh:
            for block in iter(lambda: fh.read(CHUNK_SIZE), b""):
                h.update(block)
        if h.hexdigest() != expect_md5.lower():
            sys.exit(f"md5 mismatch: {h.hexdigest()} != {expect_md5}")
        print("md5 ok", flush=True)

    part.rename(dest)
    meta_path.unlink(missing_ok=True)
    elapsed = time.monotonic() - t_start
    print(f"Done: {dest} ({human(final)}) in {elapsed / 3600:.2f} h", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("QNAP_URL"),
                   help="Direct download URL (or set QNAP_URL).")
    p.add_argument("--dest", required=True, type=Path,
                   help="Output path, e.g. data/raw/EC-EARTH3.mat")
    p.add_argument("--md5", default=None, help="Optional expected md5 checksum.")
    p.add_argument("--from-curl", metavar="FILE", default=None,
                   help="File containing a DevTools 'Copy as cURL (bash)' command; "
                        "the URL, cookies and headers are taken from it. "
                        "Use '-' to read from stdin.")
    p.add_argument("--progress", choices=("auto", "bar", "lines", "none"),
                   default="auto",
                   help="auto: tqdm bar on a terminal, log lines when redirected.")
    p.add_argument("--size", type=int, default=None, metavar="BYTES",
                   help="Exact expected file size in bytes. The file is only "
                        "declared complete at exactly this size.")
    p.add_argument("--connections", type=int, default=1, metavar="N",
                   help="Download the file over N parallel connections, each "
                        "fetching its own section (default 1). Needs a server "
                        "that supports range requests.")
    args = p.parse_args()
    if args.connections < 1:
        p.error("--connections must be at least 1")

    headers: dict[str, str] = {}
    if args.from_curl:
        raw = sys.stdin.read() if args.from_curl == "-" \
            else Path(args.from_curl).read_text()
        curl_url, headers = parse_curl(raw)
        if not curl_url:
            p.error("no http(s) URL found in that curl command")
        args.url = curl_url

    if not args.url:
        p.error("no URL given (pass --url, --from-curl, or set QNAP_URL)")

    if args.url.strip() in {"ACTUAL_DOWNLOAD_URL", "<direct download url>",
                            "URL", "https://..."}:
        p.error("that's the placeholder, not a real link — paste the URL from "
                "the share page's download button (right-click → Copy link address)")

    if not args.url.lower().startswith(("http://", "https://")):
        p.error(f"URL must start with http:// or https:// (got {args.url[:60]!r})")

    if "#" in args.url:
        p.error("that link contains a '#' fragment, which is handled by the page's "
                "JavaScript and never reaches the server — so it identifies a folder "
                "view, not a file. Open the folder in a browser and copy the link "
                "behind the file's own download icon instead.")

    _, meta_path = segment_paths(args.dest)
    if args.connections > 1 or meta_path.exists():
        # An existing plan must always be finished in segmented mode;
        # a single-stream run would append to segment 1 and corrupt it.
        download_segmented(args.url, args.dest, args.connections,
                           args.md5, args.progress, headers)
    else:
        download(args.url, args.dest, args.md5, args.progress, headers, args.size)


if __name__ == "__main__":
    main()