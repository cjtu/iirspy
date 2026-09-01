"""Geographic Lookup Tables: ship the geometry, apply it by indexing, never resample a spectrum.

A GLT is a 2-band raster on the AOI grid holding, per map pixel, the camera pixel it comes from
(band 1 = column, band 2 = absolute Scan). Applying it is advanced indexing, so every map pixel
keeps one unblended L1/L2 spectrum.

    glt = make_glt(gcps, cfg, camera_shape, scan0)
    save_glt("scene_south_glt.tif", glt, cfg, sid, group, scan0, lat_range, camera_shape)
    projected = apply_glt(l2.img.values, glt, cube_scan0, window=(r0, r1, c0, c1))

    iirs-make-glt <sid> --group south [--gcps PATH] [--out DIR]

Built by warping two index bands through `georef.project` at `Resampling.nearest`, so
`apply_glt(cube, glt)` reproduces `project(cube, ..., nearest)` bit for bit. A nearest sample reads
one source pixel, so each band keeps its own nodata with no indicator planes to unify.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from iirspy import georef

NODATA = -1

TAG_KEYS = ("sid", "group", "scan0", "lat_range", "camera_shape", "row_frame", "n_gcps", "ps", "aoi")


def make_glt(gcps, cfg, camera_shape, scan0=0, window=None) -> np.ndarray:
    """(2, y, x) int32 table for `camera_shape` under `gcps`: band 0 col, band 1 absolute Scan.

    The GCP lattice is crop-relative, so `scan0` is added back: one table then applies to the whole
    strip, to any other crop, or alongside another group's table for the same strip. The index
    bands carry no NaN, so `NODATA` means only "no camera pixel here", never "no data there".
    """
    nrow, ncol = camera_shape
    cols, rows = np.meshgrid(np.arange(ncol, dtype="float32"), np.arange(scan0, scan0 + nrow, dtype="float32"))
    warped = georef.project(np.stack([cols, rows]), gcps, cfg, resampling=Resampling.nearest, window=window)
    out = np.full(warped.shape, NODATA, "int32")
    inside = np.isfinite(warped[0]) & np.isfinite(warped[1])
    out[0][inside] = warped[0][inside].astype("int32")
    out[1][inside] = warped[1][inside].astype("int32")
    return out


def apply_glt(cube, glt, cube_scan0=0, window=None) -> np.ndarray:
    """Pull `cube` (band, row, col) onto the GLT's grid; pixels off the footprint come back NaN.

    `cube_scan0` is the absolute Scan of the cube's own row 0 -- a property of the cube in hand,
    not of the solve that built the table.
    """
    cube = np.asarray(cube)
    if cube.ndim == 2:
        cube = cube[None]
    if window is not None:
        r0, r1, c0, c1 = window
        glt = glt[:, r0:r1, c0:c1]
    col = glt[0]
    inside = col >= 0
    row = glt[1] - cube_scan0
    if inside.any() and (
        row[inside].min() < 0 or row[inside].max() >= cube.shape[1] or col[inside].max() >= cube.shape[2]
    ):
        raise ValueError(
            f"glt wants camera scans {glt[1][inside].min()}-{glt[1][inside].max()} and columns up to "
            f"{col[inside].max()}, but the cube is {cube.shape[1]} rows x {cube.shape[2]} cols starting at "
            f"scan {cube_scan0} -- wrong scene, or the wrong cube_scan0"
        )
    out = np.full((cube.shape[0], *col.shape), np.nan, "float32")
    out[:, inside] = cube[:, row[inside], col[inside]]
    return out


def save_glt(fout, glt, cfg, sid, group, scan0, lat_range, camera_shape, window=None, n_gcps=0) -> Path:
    """Write a 2-band int32 COG."""
    (ny, nx), tr = georef.window_of(cfg, window)
    tags = {
        "sid": sid,
        "group": group,
        "scan0": scan0,
        "lat_range": list(lat_range),
        "camera_shape": list(camera_shape),
        "row_frame": "absolute_scan",
        "n_gcps": n_gcps,
        "ps": cfg.ps,
        "aoi": list(cfg.aoi),
    }
    profile = {
        "driver": "COG",
        "height": ny,
        "width": nx,
        "count": 2,
        "dtype": "int32",
        "nodata": NODATA,
        "crs": georef.stereo_crs(cfg.pole),
        "transform": tr,
        "BLOCKSIZE": 256,
        "COMPRESS": "DEFLATE",
        "PREDICTOR": 2,  # both bands are near-linear ramps along a scanline
        "OVERVIEWS": "NONE",  # an averaged pixel index is meaningless
        "BIGTIFF": "YES",
    }
    with rasterio.open(fout, "w", **profile) as dst:
        dst.write(glt)
        dst.update_tags(**{k: json.dumps(v) for k, v in tags.items()})
        dst.set_band_description(1, "camera_col")
        dst.set_band_description(2, "camera_row")
    return Path(fout)


def read_glt(path) -> tuple[np.ndarray, dict]:
    """(2, y, x) int32 table and its tags."""
    with rasterio.open(path) as src:
        raw = src.tags()
        return src.read(), {k: json.loads(raw[k]) for k in TAG_KEYS if k in raw}


def save_gcp_vrt(product, gcps, crs, fout=None) -> Path:
    """VRT beside a camera-space product carrying its GCPs, so GDAL can georeference it.

    A sidecar because a GeoTIFF holds either a geotransform or GCPs, and the product needs its
    transform for `scan0`. `gdalwarp -tps -et 0 -r near` on it reproduces `apply_glt`.
    """
    import xml.etree.ElementTree as ET

    product = Path(product)
    with rasterio.open(product) as src:
        count, ny, nx, dtype = src.count, src.height, src.width, src.dtypes[0]
    wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)

    root = ET.Element("VRTDataset", rasterXSize=str(nx), rasterYSize=str(ny))
    lst = ET.SubElement(root, "GCPList", Projection=wkt)
    for i, g in enumerate(gcps):
        ET.SubElement(lst, "GCP", Id=str(i), Pixel=repr(g.col), Line=repr(g.row), X=repr(g.x), Y=repr(g.y))
    for b in range(1, count + 1):
        band = ET.SubElement(root, "VRTRasterBand", dataType=dtype.capitalize(), band=str(b))
        ET.SubElement(band, "NoDataValue").text = "nan"
        src_el = ET.SubElement(band, "SimpleSource")
        ET.SubElement(src_el, "SourceFilename", relativeToVRT="1").text = product.name
        ET.SubElement(src_el, "SourceBand").text = str(b)
        for tag in ("SrcRect", "DstRect"):
            ET.SubElement(src_el, tag, xOff="0", yOff="0", xSize=str(nx), ySize=str(ny))

    out = Path(fout) if fout else product.with_suffix(".vrt")
    out.write_bytes(ET.tostring(root))
    return out


def scene_glt(sid, group, gcps, cfg, scan0, glt_dir, overwrite=False) -> Path:
    """The scene's GLT, built if it is not there yet. `overwrite` for GCPs that just changed."""
    f = Path(glt_dir) / f"{sid}_{group}_glt.tif"
    if f.exists() and not overwrite:
        return f
    f.parent.mkdir(parents=True, exist_ok=True)
    cam = (int(max(g.row for g in gcps)) + 1, int(max(g.col for g in gcps)) + 1)
    table = make_glt(gcps, cfg, cam, scan0=scan0)
    return save_glt(f, table, cfg, sid, group, scan0, cfg.lat_band, cam, n_gcps=len(gcps))


def _parser():
    import argparse

    from iirspy import chunks as ck

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sid", help="scene id, e.g. 20210103T1829495344")
    ap.add_argument("--group", required=True, choices=list(ck.GROUPS))
    ap.add_argument("--gcps", default=None, help="merged .gcps (default the iirs-solve-scene --keep layout)")
    ap.add_argument("--out", default=".", help="directory to write <sid>_<group>_glt.tif into")
    ap.add_argument(
        "--scan0",
        type=int,
        default=0,
        help="first Scan of the L1 crop the GCP rows are relative to",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    from dataclasses import replace

    from iirspy import chunks as ck
    from iirspy import refl

    args = _parser().parse_args(argv)
    fgcps = (
        Path(args.gcps)
        if args.gcps
        else Path.home()
        / "data"
        / "iirs"
        / "gcps"
        / f"{args.sid}_{args.group}"
        / f"{args.sid}_{args.group}_merged.gcps"
    )
    if not fgcps.is_file() or fgcps.stat().st_size == 0:
        sys.exit(f"{fgcps} missing or empty -- {args.sid} {args.group} is not solved yet")

    gcps, aoi = refl._gcps_and_aoi(fgcps)
    lat_range = ck.l1_lat_range(args.group)
    cfg = replace(ck.chunk_cfg(args.group), aoi=aoi, lat_band=lat_range)
    # The lattice spans the camera region the TPS was fitted over, so no cube is needed.
    shape = (int(max(g.row for g in gcps)) + 1, int(max(g.col for g in gcps)) + 1)
    glt = make_glt(gcps, cfg, shape, scan0=args.scan0)

    out = Path(args.out)
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
