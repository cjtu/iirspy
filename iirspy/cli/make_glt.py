from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from iirspy.cli._run import paths_from_env
from iirspy.georef import chunks as ck
from iirspy.georef.georef import gcps_and_aoi, make_glt, save_glt


def _glt_parser():
    import argparse

    ap = argparse.ArgumentParser(description="Build a scene's GLT from its merged GCPs.")
    ap.add_argument("sid", help="scene id, e.g. 20210103T1829495344")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--gcps", default=None, help="merged .gcps (default the iirs-solve-scene --keep layout)")
    ap.add_argument(
        "--out",
        default=None,
        help="directory to write <sid>_<group>_glt.tif into (default alongside the GCPs, "
        "geometry/recalibrated/<day> under IIRS_RECAL_ROOT)",
    )
    ap.add_argument(
        "--scan0",
        type=int,
        default=0,
        help="first Scan of the L1 crop the GCP rows are relative to",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _glt_parser().parse_args(argv)
    paths = paths_from_env()
    fgcps = Path(args.gcps) if args.gcps else ck.merged_gcps_path(args.sid, args.group, paths)
    if not fgcps.is_file() or fgcps.stat().st_size == 0:
        sys.exit(f"{fgcps} missing or empty -- {args.sid} {args.group} is not solved yet")

    try:
        gcps, aoi = gcps_and_aoi(fgcps)
    except ValueError as e:
        sys.exit(str(e))
    lat_range = ck.l1_lat_range(args.group)
    cfg = replace(ck.chunk_cfg(args.group, paths.dem_roots), aoi=aoi, lat_band=lat_range)
    # The lattice spans the camera region the TPS was fitted over, so no cube is needed.
    shape = (int(max(g.row for g in gcps)) + 1, int(max(g.col for g in gcps)) + 1)
    glt = make_glt(gcps, cfg, shape, scan0=args.scan0)

    out = Path(args.out) if args.out else ck.recal_dir(args.sid, paths)
    out.mkdir(parents=True, exist_ok=True)
    f = save_glt(
        out / f"{args.sid}_{args.group}_glt.tif",
        glt,
        cfg,
        args.sid,
        args.group,
        args.scan0,
        lat_range,
        shape,
        n_gcps=len(gcps),
    )
    valid = int((glt[0] >= 0).sum())
    print(
        f"{f} {glt.shape[1]}x{glt.shape[2]} px, {valid / glt[0].size:.1%} inside footprint, "
        f"{f.stat().st_size / 1e6:.1f} MB, camera {shape}, {len(gcps)} gcps",
        flush=True,
    )


if __name__ == "__main__":
    main()
