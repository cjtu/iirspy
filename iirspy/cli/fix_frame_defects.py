"""Repair frame defects in an ISSDC nri or nci cube and write it in nci camera order (`iirspy.l1.frame_defects`).

The frame-defect table comes from that cube, built here when absent. Frame defects are an L1 step, so an ndi is never
repaired: some are already re-timed by ISSDC, and our own L2 is built from a repaired L1.
"""

import argparse
from pathlib import Path

from iirspy import io
from iirspy.cli._run import frame_defects_path
from iirspy.iirs import L0, L1
from iirspy.l1 import frame_defects

LEVELS = {"nri": L0, "nci": L1}


def main(argv=None):
    paths = io.Paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid")
    ap.add_argument("--product", required=True, choices=LEVELS, help="level of the input cube")
    ap.add_argument("--directory", default=None, help="ISSDC tree holding it (default IIRS_ARCHIVE)")
    ap.add_argument("--table", help="frame-defect table (default: the strip's <sid>_frame_defects.csv beside its GCPs)")
    ap.add_argument("--fill-frame-defects", action="store_true", help="rebuild the lost frames instead of leaving NaN")
    ap.add_argument("--out", required=True, help="output .tif")
    args = ap.parse_args(argv)
    cube = LEVELS[args.product](args.sid, args.directory or paths.archive)
    table = Path(args.table or frame_defects_path(args.sid, paths))
    if not table.exists():
        frame_defects.build_table(cube.qub, table)
    cube.fix_frame_defects(table, args.fill_frame_defects)
    print(cube.save(args.out))


if __name__ == "__main__":
    main()
