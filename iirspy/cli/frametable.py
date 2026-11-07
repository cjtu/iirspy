"""Build the frame-defect tables of raw nri zips and nothing else (see `iirspy.l1.frames`), named as frame-table db
members (`frames.db_name`: `<sid>_<data center>_frames_offsets.csv`), so `--out` zips straight into the db.

Each argument is a strip id (its preferred data center's zip is looked up in `--zips`) or a path to a zip. Tables already in `--out` are kept
unless `--force`, so a timed-out batch resumes where it stopped. A failing scene is reported and skipped; the exit
status is 1 if any failed.
"""

import argparse
import sys
import time
from pathlib import Path

from iirspy import io
from iirspy.l1 import frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenes", nargs="+", help="strip ids or nri zip paths")
    ap.add_argument("--out", required=True, type=Path, help="directory for the tables")
    ap.add_argument("--zips", type=Path, default=None, help="directory of nri zips (default IIRS_ARCHIVE/zips)")
    ap.add_argument("--force", action="store_true", help="rebuild tables that already exist")
    args = ap.parse_args(argv)
    zips = args.zips or io.Paths().archive / "zips"
    failed = 0
    for s in args.scenes:
        zp = (
            Path(s)
            if s.endswith(".zip")
            else min(zips.glob(f"ch2_iir_nri_{s}_*.zip"), key=io.collect_order, default=None)
        )
        if zp is None or not zp.exists():
            failed += 1
            print(f"{s}: FAILED no nri zip in {zips}", flush=True)
            continue
        name, fcsv = zp.name, args.out / frames.db_name(zp)
        if fcsv.exists() and not args.force:
            print(f"{name}: have table", flush=True)
            continue
        t0 = time.time()
        try:
            frames.build_table(zp, fcsv)
        except Exception as e:  # one bad scene must not sink the batch
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        print(f"{name}: {time.time() - t0:.0f}s {zp.stat().st_size / 1e9:.2f} GB", flush=True)
    sys.exit(bool(failed))


if __name__ == "__main__":
    main()
