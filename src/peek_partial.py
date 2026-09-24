#!/usr/bin/env python3
"""Look inside a partially downloaded MATLAB v7.3 (HDF5) file.

Lists the variables (name, shape, type, chunking, compression) and reports how
much of each one is already inside the downloaded part.

READ-ONLY: the .part file is opened with 'rb' and never modified, so it is safe
to run while download_data.py is writing to it. It reads metadata only, no bulk
data, so memory use stays small.

How it works: HDF5 refuses to open a file that is shorter than the size stored
in its header. This script gives h5py a wrapper that reports the full expected
size and returns nothing for bytes that have not arrived yet (h5py fills them
with zeros). So anything beyond the downloaded part either fails to read
(reported below) or reads as zeros: only trust data in chunks this report marks
as downloaded.

    .venv/bin/python src/peek_partial.py data/raw/historical/EC-EARTH3.mat.part
"""
import argparse
import io
import os
import sys

import h5py

SIZES = {"historical": 59_982_783_406, "ssp126": 60_656_201_821,
         "ssp585": 60_874_380_344}
MAX_CHILDREN = 40      # per group; MATLAB cell arrays can have thousands of entries
SAMPLE_IF_OVER = 20_000  # without chunk_iter, scanning more chunks than this is slow


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def short(exc: BaseException) -> str:
    return str(exc).splitlines()[0][:110] if str(exc) else type(exc).__name__


class PartialFile(io.RawIOBase):
    """Read-only view of a partial file that pretends to be full size."""

    def __init__(self, path: str, full_size: int):
        self._f = open(path, "rb", buffering=0)
        self.avail = os.fstat(self._f.fileno()).st_size   # snapshot at open
        self.full = full_size
        self.pos = 0
        self.missing_reads = 0     # reads that touched not-yet-downloaded bytes

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:                                  # SEEK_END: pretend to be complete
            self.pos = self.full + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, buf):
        view = memoryview(buf).cast("B")
        want = len(view)
        n = max(0, min(want, self.avail - self.pos))
        got = 0
        if n:
            self._f.seek(self.pos)
            got = self._f.readinto(view[:n]) or 0
        if got < want:
            self.missing_reads += 1
        self.pos += got
        return got

    def close(self):
        if not self.closed:
            self._f.close()
        super().close()


def matlab_class(obj) -> str:
    try:
        c = obj.attrs.get("MATLAB_class")
    except Exception:
        return "?"
    if isinstance(c, bytes):
        c = c.decode(errors="replace")
    return str(c) if c is not None else ""


def availability(dset, avail: int) -> str:
    dsid = dset.id
    if dset.chunks is None:                    # contiguous storage
        try:
            off, size = dsid.get_offset(), dsid.get_storage_size()
        except Exception as e:
            return f"storage info not readable yet ({short(e)})"
        if off is None or size == 0:
            return "no data stored"
        have = max(0, min(off + size, avail) - off)
        return (f"contiguous at byte {off:,}: {100 * have / size:.0f}% downloaded "
                f"({human(have)} of {human(size)})")

    shape, cshape = dset.shape, dset.chunks
    st = {"n": 0, "done": 0, "nb": 0, "db": 0, "lo": None, "hi": None}

    def visit(info):
        st["n"] += 1
        st["nb"] += info.size
        if info.byte_offset + info.size <= avail:
            st["done"] += 1
            st["db"] += info.size
            o = tuple(info.chunk_offset)
            e = tuple(min(a + c, s) for a, c, s in zip(o, cshape, shape))
            st["lo"] = o if st["lo"] is None else tuple(map(min, st["lo"], o))
            st["hi"] = e if st["hi"] is None else tuple(map(max, st["hi"], e))

    approx = ""
    try:
        if hasattr(dsid, "chunk_iter"):
            dsid.chunk_iter(visit)
        else:
            total = dsid.get_num_chunks()
            if total > SAMPLE_IF_OVER:
                step = total // 500
                idx = range(0, total, step)
                approx = f" (estimated from {len(idx)} of {total:,} chunks)"
            else:
                idx = range(total)
            for i in idx:
                visit(dsid.get_chunk_info(i))
    except Exception as e:
        return f"chunk index not readable yet, so treat as 0% downloaded ({short(e)})"

    if st["n"] == 0:
        return "no chunks stored"
    pct = 100 * st["db"] / st["nb"]
    msg = (f"{pct:.0f}% downloaded{approx}: {st['done']:,} of {st['n']:,} chunks, "
           f"{human(st['db'])} of {human(st['nb'])} compressed")
    if st["done"] and not approx:
        box = ", ".join(f"{a}:{b}" for a, b in zip(st["lo"], st["hi"]))
        msg += f"\n{'':8}downloaded chunks lie within h5py index range [{box}] (may have gaps)"
    return msg


def describe(dset, avail: int, pad: str) -> None:
    shape = dset.shape
    mat = "x".join(str(d) for d in reversed(shape)) if shape else "scalar"
    comp = dset.compression or "none"
    if dset.compression_opts is not None:
        comp += f"({dset.compression_opts})"
    print(f"{pad}{dset.name.rsplit('/', 1)[-1]}: MATLAB {matlab_class(dset) or '?'} "
          f"{mat}  | h5py shape {shape} {dset.dtype}")
    print(f"{pad}      chunks {dset.chunks}, compression {comp}")
    print(f"{pad}      {availability(dset, avail)}")


def walk(group, avail: int, depth: int = 0) -> None:
    pad = "  " * depth
    try:
        names = list(group.keys())
    except Exception as e:
        print(f"{pad}  [contents not readable yet: {short(e)}]")
        return
    for k, name in enumerate(names):
        if k == MAX_CHILDREN:
            print(f"{pad}  ... and {len(names) - MAX_CHILDREN} more")
            break
        try:
            obj = group[name]
        except Exception as e:
            print(f"{pad}{name}: [not readable yet: {short(e)}]")
            continue
        if isinstance(obj, h5py.Group):
            print(f"{pad}{name}/  (MATLAB {matlab_class(obj) or 'group'})")
            walk(obj, avail, depth + 1)
        elif isinstance(obj, h5py.Dataset):
            try:
                describe(obj, avail, pad)
            except Exception as e:
                print(f"{pad}{name}: [not readable yet: {short(e)}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="the .part (or complete .mat) file")
    ap.add_argument("--size", type=int, help="expected full size in bytes "
                    "(guessed from the scenario folder if omitted)")
    args = ap.parse_args()

    size = args.size or next((v for k, v in SIZES.items()
                              if f"/{k}/" in os.path.abspath(args.path)), None)
    if not size:
        sys.exit("Could not guess the expected size from the path; pass --size.")

    pf = PartialFile(args.path, size)
    print(f"File:       {args.path}")
    print(f"Downloaded: {human(pf.avail)} of {human(size)} "
          f"({100 * pf.avail / size:.1f}%)")
    print(f"h5py {h5py.__version__}, HDF5 {h5py.version.hdf5_version}\n")

    try:
        h5 = h5py.File(pf, "r")
    except Exception as e:
        pf.close()
        sys.exit(f"h5py could not open it: {short(e)}\n"
                 "The file's top-level metadata is probably not downloaded yet.")
    with h5:
        walk(h5, pf.avail)
    pf.close()

    print("\nNotes:")
    print("- 'MATLAB AxB' is the size as MATLAB shows it; h5py sees the dimensions "
          "reversed.")
    if pf.missing_reads:
        print(f"- {pf.missing_reads} metadata reads reached past the downloaded part, "
              "so some entries above may be incomplete.")
    print("- Only data in chunks marked as downloaded is real; the rest reads as "
          "zeros or errors.")


if __name__ == "__main__":
    main()
