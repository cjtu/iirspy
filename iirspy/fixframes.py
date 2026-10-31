"""Repair frame defects in an ISSDC nri, nci or ndi cube and write it in nci camera order (see `iirspy.frames`).

The frame table comes from the nri or nci cube, built here when absent; an ndi needs it from an earlier run.
"""

import argparse
from pathlib import Path

from iirspy import chunks as ck
from iirspy import frames, solve
from iirspy.iirs import L0, L1, L2

LEVELS = {"nri": L0, "nci": L1, "ndi": L2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid")
    ap.add_argument("--product", required=True, choices=LEVELS, help="level of the input cube")
    ap.add_argument("--directory", default=ck.ARCHIVE, help="ISSDC tree holding it (default IIRS_ARCHIVE)")
    ap.add_argument("--table", help="frame table (default: the strip's <sid>_frames_offsets.csv beside its GCPs)")
    ap.add_argument("--fill-frames", action="store_true", help="rebuild the lost frames instead of leaving NaN")
    ap.add_argument("--out", required=True, help="output .tif")
    args = ap.parse_args(argv)
    cube = LEVELS[args.product](args.sid, args.directory)
    table = Path(args.table or solve.frame_offsets_path(args.sid))
    if args.product != "ndi" and not table.exists():
        frames.build_table(cube.qub, table)
    cube.fix_frames(table, args.fill_frames)
    print(cube.save(args.out))


if __name__ == "__main__":
    main()
