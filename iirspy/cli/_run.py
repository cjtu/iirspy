"""Shared plumbing of the `iirs-georef-solve` and `iirs-compute-refl` CLIs: work dir, staging, L1 build, outputs.

A run's inputs come from the `IIRS_*` env vars (`io.Paths`); `stage_inputs` pulls only the bands it needs from the
nri zip, so a local run and a cluster array task behave identically.
"""

from __future__ import annotations

import argparse
import datetime
import errno
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import rasterio

from iirspy import instrument, io
from iirspy.georef import ancillary
from iirspy.georef import chunks as ck
from iirspy.io import ncpu
from iirspy.l1.config import CALIB, L1Config
from iirspy.l1.frame_defects import table_key as defects_key
from iirspy.l2.config import L2Config

log = logging.getLogger(__name__)
# One table per raw zip, members named by `frame_defects.db_name`.
FRAME_DEFECT_DB = "geometry/recalibrated/frame_defect_tables.zip"

CALIBRATE_KWS = {
    "dark": "empirical",
    "flat": "empirical",
    "interp_bands": "linear",
    "interp_spatial": True,
    "bad_pixel_mask": True,
}


@dataclass(frozen=True)
class Job:
    sid: str
    group: str
    zip: Path
    stage: Path
    logfile: Path
    paths: io.Paths
    work: Path
    frame_defects: Path | None = None


def open_run(sid: str, group: str, out: str | None, step: str, paths: io.Paths) -> Job:
    """Locate the scene's zip and prepare the work dir, `<sid>_<group>_<step>.log` and stage."""
    if not paths.archive.exists():
        sys.exit(f"{paths.archive} missing -- archive not mounted, set IIRS_ARCHIVE")
    zip_path = ck.nri_zip(sid, paths)
    if zip_path is None:
        sys.exit(f"no nri zip for {sid} under {paths.archive / 'zips'}")
    work = Path(out) if out else Path.cwd() / "runs" / f"{sid}_{group}"
    logfile = work / f"{sid}_{group}_{step}.log"
    work.mkdir(parents=True, exist_ok=True)
    stage = Path(os.environ.get("IIRS_STAGE") or work.parent / "stage")
    stage.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(logfile, mode="w")],
        force=True,
    )
    return Job(sid, group, zip_path, stage, logfile, paths, work)


@contextmanager
def timed(name: str):
    t0 = time.monotonic()
    yield
    log.info("%s: %.0fs", name, time.monotonic() - t0)


def log_provenance() -> None:
    log.info("provenance: %s", json.dumps(io.provenance(), default=str))


def stage_inputs(job: Job, day: str, bands: list[int]) -> Path:
    """Extract the zip and assemble the one-root PDS tree `L0` reads. Returns the raw cube's path.

    The zip brings every nri member (cube bands, spm, oat, xml); only the nci products it lacks are
    copied in. The geometry csv comes from every `paths.anc_roots`, not just the archive: without it
    `l1.csv` is None and the crop dies in `parse_geom` after the whole cube has calibrated.
    """
    raw_qub = job.stage / f"data/raw/{day}/{job.zip.stem}.qub"
    if not raw_qub.exists():
        log.info(f"extracting {job.zip.name} ({len(bands)} bands) -> {job.stage}")
        with timed("extraction"):
            io.extract(job.zip, job.stage, bands=bands)
    for sub in (
        f"geometry/calibrated/{day}",
        f"miscellaneous/calibrated/{day}",
        f"data/calibrated/{day}",
    ):
        src_dir = job.paths.archive / sub
        if not src_dir.exists():
            continue
        dst_dir = job.stage / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.glob(f"*{job.sid}*"):
            if not (dst_dir / f.name).exists():
                shutil.copyfile(f, dst_dir / f.name)
    fgeom = ck.ancillary(job.sid, job.paths)["geometry/calibrated"]
    if fgeom is not None:
        dst = job.stage / "geometry/calibrated" / day / fgeom.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(fgeom, dst)
    return raw_qub


def build_l1(
    job: Job,
    lat_range: tuple[float, float],
    bands: list[int],
    ftif_out: Path,
    calibrate_kwargs: dict | None = None,
    out_bands: list[int] | None = None,
    chunk_y: int = 1024,
    keep_raw: bool = False,
    l1_config=None,
) -> tuple[Path, int, tuple[float, float]]:
    """Extract the nri zip and calibrate it, cropped to `lat_range`.

    The zip's internal layout already matches the archive's PDS tree, so it extracts in place.
    Returns the L1 path, the crop's first `Scan` value (row 0's scan id, for turning a chunk's
    scan_lo/scan_hi into row indices), and the latitudes the returned file actually covers, which
    for a cache hit may be a wider earlier build. `bands` is part of the cache key -- a cube staged
    for one band list is never reused for a different one, e.g. a single-band solve cache can't
    satisfy a full-spectrum refl run.

    `out_bands` narrows what actually gets *written*, after `bands` has done its job of giving the
    dark/flat/smile search enough of the spectrum to work with -- a chunk solve only ever reads one
    band back out (`read_band(ftif, cfg.band)`), so writing all of `bands` to disk for that case is
    pure I/O nobody uses. None (default, what `iirspy.cli.refl` wants) writes every calibrated band.

    l1_config (an `L1Config`) replaces CALIBRATE_KWS and calibrate_kwargs outright (`iirs-compute-refl`);
    its full resolved fields, plus a calib LUT's content hash, are the cache key.

    calibrate_kwargs overrides CALIBRATE_KWS (e.g. interp_bands=None, exclude_wl=[...]) for a
    one-off reprocessing run without touching the production default -- it's part of the cache key
    (via the `.meta.json` sidecar) same as `lat_range`/`bands`, so a rebuild with different
    calibration settings into the same `ftif_out` never returns the other run's cube. A cached
    file with no recorded `calibrate_kwargs` (written before this check existed) is treated as
    unknown, not as "assume production defaults", and rebuilt once to backfill the tag. `chunk_y`
    is the dask y-chunk the cube is read and written in, and so also the write block size.

    keep_raw : bool
        Skip deleting the extracted raw `.qub`/`.hdr` after calibration. Default False (production
        behaviour: extract, calibrate, delete -- C: is the scarce drive). Set True for iterative
        reprocessing of the same scene (e.g. re-deriving L1 with different calibrate_kwargs) so
        `stage_inputs` can reuse the already-extracted raw cube instead of re-reading the zip
        (which usually lives under `paths.archive`, i.e. ddata) on every call.
    """
    # Ancillary only (spm, oat, xml, csv -- excludes the qub by default): a few MB, so cheap enough
    # to always re-run even on an L1 cache hit. Without this, a cache hit skips `stage_inputs`
    # below and the stage never gets the spm the zip carries, since it does not live in the archive.
    io.extract(job.zip, job.stage)
    # The nri zip's own layout only ever yields .../raw, but iirspy.L1 (level=1) always looks
    # under .../calibrated -- for spm (calibrate()'s FileNotFoundError check) and for the xml
    # label (from_xarray's solar_inc/solar_az lookup, iirs.py `paths.get("qub"/"xml")` at
    # level=1). Mirror both so a stage-only run (no full nci archive copy) still satisfies them.
    day = job.sid[:8]
    for root, ext in (("miscellaneous", "spm"), ("data", "xml")):
        src_dir = job.stage / root / "raw" / day
        dst_dir = job.stage / root / "calibrated" / day
        for fsrc in src_dir.glob(f"*{job.sid}*.{ext}"):
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not (dst_dir / fsrc.name).exists():
                shutil.copyfile(fsrc, dst_dir / fsrc.name)

    saved_bands = out_bands if out_bands is not None else bands
    effective_calibrate_kws = {**CALIBRATE_KWS, "frame_defects": str(job.frame_defects), **(calibrate_kwargs or {})}
    if l1_config is not None:
        effective_calibrate_kws = {"l1_config": l1_config.to_attrs(), "frame_defects": str(job.frame_defects)}
        if l1_config.calib not in CALIB:
            effective_calibrate_kws["calib_sha256"] = hashlib.sha256(Path(l1_config.calib).read_bytes()).hexdigest()
    # Crop with the geometry csv the solve's GCP lattice uses: a sid can carry several g_grd csvs
    # (d18/n18) whose latitudes differ, so the label's own csv can cut the cube rows differently.
    fgeom = ck.ancillary(job.sid, job.paths)["geometry/calibrated"]
    crop_geom = fgeom.name if fgeom is not None else "label"
    fmeta = ftif_out.with_suffix(".meta.json")
    if ftif_out.exists() and fmeta.exists():
        meta = json.loads(fmeta.read_text())
        cached_range = tuple(meta["lat_range"])
        if (
            cached_range[0] <= lat_range[0]
            and lat_range[1] <= cached_range[1]
            and meta.get("bands") == saved_bands
            and "calibrate_kwargs" in meta
            and meta["calibrate_kwargs"] == effective_calibrate_kws
            and meta.get("frame_defects") == defects_key(job.frame_defects)
            and meta.get("crop_geom") == crop_geom
        ):
            log.info(f"L1 already built: {ftif_out} (covers {cached_range}, requested {lat_range})")
            return ftif_out, meta["scan0"], cached_range
        log.info(
            f"cached L1 {cached_range}/bands={meta.get('bands')}/"
            f"calibrate_kwargs={meta.get('calibrate_kwargs', '<unrecorded>')} does not match requested "
            f"{lat_range}/{bands}/{effective_calibrate_kws} -- rebuilding"
        )
    from iirspy import L0

    with timed("staging"):
        raw_qub = stage_inputs(job, day, bands)

    # `chunk_y` sets both the dask y-chunk and, through `_row_block`, the write block.
    with timed("L0 read + calibrate (graph build; the compute lands in the write)"):
        l0 = L0(job.sid, job.stage, chunk={"band": -1, "y": chunk_y, "x": -1})
        l1 = (
            l0.calibrate(**effective_calibrate_kws)
            if l1_config is None
            else l0.calibrate(l1_config, frame_defects=str(job.frame_defects))
        )
    # l1.img already holds exactly `bands` -- that's what _stage_inputs staged.
    _, xyext = ancillary.parse_geom(fgeom or l1.csv, latlonextent=(-180, 180, *lat_range))
    ymin, ymax = xyext[2], xyext[3]
    log.info(f"L1 crop lat={lat_range}: scan {ymin}-{ymax} ({ymax - ymin} rows)")
    l1.img = l1.img.sel(y=slice(ymin, ymax))
    l1.img = l1.img / instrument.RAD_NATIVE_SCALE
    l1.img.attrs["units"] = "1000 mW/cm^2/sr/um"
    if out_bands is not None:
        l1.img = l1.img.sel(band=out_bands)
    with timed(f"L1 write ({l1.img.shape[0]} band x {l1.img.shape[1]} rows, chunk_y={chunk_y})"):
        l1.save(str(ftif_out))
    fmeta.write_text(
        json.dumps({
            "scan0": int(ymin),
            "lat_range": list(lat_range),
            "bands": saved_bands,
            "calibrate_kwargs": effective_calibrate_kws,
            "frame_defects": defects_key(job.frame_defects),
            "crop_geom": crop_geom,
        })
    )
    if not keep_raw:
        raw_qub.unlink(missing_ok=True)
        raw_qub.with_suffix(".hdr").unlink(missing_ok=True)
    log.info(f"L1 saved: {ftif_out}")
    return ftif_out, int(ymin), lat_range


def build_l1_nci(
    job: Job,
    lat_range: tuple[float, float],
    bands: list[int],
    ftif_out: Path,
    chunk_y: int = 1024,
    fix_frame_defects: bool = False,
    fill_frame_defects: bool = False,
) -> tuple[Path, int, tuple[float, float]]:
    """`build_l1`'s counterpart for ISSDC's own L1: the nci cube, cropped to `lat_range`, written in the same
    layout (native radiance units, `bands` only, `.meta.json` beside it) so everything after it is shared.

    The cube is unzipped into the ISSDC tree under `paths.archive` if needed (`ck.ensure_nci`). No empirical dark/flat
    or band repair: ISSDC's calibration as shipped, plus `IIRSData.fix_frame_defects` with `job.frame_defects` if asked
    (`fill_frame_defects` implies `fix_frame_defects`). Returns the L1 path, row 0's `Scan` and `lat_range`.
    """
    from iirspy import L1

    fix_frame_defects = fix_frame_defects or fill_frame_defects
    fmeta = ftif_out.with_suffix(".meta.json")
    if ftif_out.exists() and fmeta.exists():
        meta = json.loads(fmeta.read_text())
        if (
            meta.get("source") == "issdc-nci"
            and meta.get("bands") == bands
            and tuple(meta["lat_range"]) == tuple(lat_range)
            and meta.get("frame_defects")
            == [fix_frame_defects, fill_frame_defects, defects_key(job.frame_defects) if fix_frame_defects else None]
        ):
            log.info(f"nci L1 already built: {ftif_out}")
            return ftif_out, meta["scan0"], lat_range
    with timed("nci unzip"):
        ck.ensure_nci(job.sid, job.paths)
    with timed("nci L1 read + crop"):
        l1 = L1(
            job.sid, job.paths.archive, lonlatextent=(-180, 180, *lat_range), chunk={"band": -1, "y": chunk_y, "x": -1}
        )
        scan0 = int(l1.extent[2])
        l1.img = l1.img.sel(band=bands) / instrument.RAD_NATIVE_SCALE
        l1.img.attrs["units"] = "1000 mW/cm^2/sr/um"
        if fix_frame_defects:
            l1.fix_frame_defects(job.frame_defects, fill_frame_defects)
    with timed(f"nci L1 write ({l1.img.shape[0]} band x {l1.img.shape[1]} rows)"):
        l1.save(str(ftif_out), snr_sidecar=False, empirical_sidecar=False)
    defects_meta = [
        fix_frame_defects,
        fill_frame_defects,
        defects_key(job.frame_defects) if fix_frame_defects else None,
    ]
    fmeta.write_text(
        json.dumps({
            "scan0": scan0,
            "lat_range": list(lat_range),
            "bands": bands,
            "source": "issdc-nci",
            "frame_defects": defects_meta,
        })
    )
    log.info(f"nci L1 saved: {ftif_out}")
    return ftif_out, scan0, lat_range


def frame_defects_path(sid: str, paths: io.Paths, dest: Path | None = None) -> Path:
    """`sid`'s frame-defect table: one per strip, beside its GCPs and `_L1_b*.tif` in `dest` (default
    `ck.recal_dir`)."""
    return Path(dest or ck.recal_dir(sid, paths)) / f"{sid}_frame_defects.csv"


def ensure_frame_defects(job: Job, dest: Path | None = None) -> Path:
    """`job.sid`'s frame-defect table, cached for every later task and group: copied from the archive's frame-defect db
    (`geometry/recalibrated/frame_defect_tables.zip`), else built from the whole zip (~1 min, streamed, not staged).
    """
    from iirspy.l1 import frame_defects

    f = frame_defects_path(job.sid, job.paths, dest)
    if f.exists() or frame_defects.from_db(job.paths.archive / FRAME_DEFECT_DB, job.zip, f):
        return f
    log.info(f"frame-defect table missing, building from {job.zip.name} -> {f}")
    with timed("frame table"):
        frame_defects.build_table(job.zip, f)
    return f


def predates_frame_defects(fgcps: Path, fdefects: Path) -> bool:
    """Whether `fgcps` was solved before `fdefects` existed, i.e. on a cube whose groups were not re-timed."""
    return not fdefects.exists() or fgcps.stat().st_mtime < fdefects.stat().st_mtime


def clean_work(work: Path, clean: bool) -> None:
    if clean:
        shutil.rmtree(work, ignore_errors=True)
        print(f"removed work dir {work}", flush=True)


def keep(work: Path, dest_dir: str | None, *files: Path) -> None:
    """Copy products out of the work dir into `dest_dir` (a no-op without one)."""
    if not dest_dir:
        return
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / f.name)
    log.info(f"kept {len(files)} file(s) -> {dest}")


def is_disk_full(exc: BaseException) -> bool:
    """Whether `exc` is a quota / ENOSPC failure, as an OSError errno or as a message-only GDAL error."""
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        return True
    msg = str(exc).lower()
    return "disk quota exceeded" in msg or "no space left on device" in msg


def file_info(f, sha=False) -> dict:
    """Path and mtime (UTC) of an input, plus its sha256 when `sha`."""
    f = Path(f)
    mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime, datetime.UTC).isoformat(timespec="seconds")
    return {"path": str(f), "mtime": mtime, **({"sha256": hashlib.sha256(f.read_bytes()).hexdigest()} if sha else {})}


def resource_report(img) -> str:
    """Whether the cube's y-blocks keep `ncpu` dask threads busy, and the peak RSS."""
    import resource

    n, ny, nx = img.shape
    ychunks = img.chunks[img.dims.index("y")] if img.chunks is not None else (ny,)
    block_mb = n * max(ychunks) * nx * 4 / 2**20
    busy = "ok" if len(ychunks) >= ncpu() else "STARVED: fewer blocks than cores"
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return f"L1 {img.shape} in {len(ychunks)} y-blocks of <={block_mb:.0f} MB for {ncpu()} workers ({busy}); peak RSS {peak_mb:.0f} MB"


def solve_quality(fgcps: Path) -> dict:
    """converged/corr/n_merged_gcps/overlap agreement from the solve's own summary json, if kept
    next to the GCPs -- so a suspect refl product points at the registration it was built on."""
    fsummary = fgcps.parent / f"{fgcps.stem}_solve_summary.json"
    if not fsummary.exists():
        return {}
    s = json.loads(fsummary.read_text())
    fits = s.get("per_chunk_fit", [])
    return {
        "converged": all(f.get("converged") for f in fits) if fits else None,
        "corr": [f.get("corr") for f in fits],
        "n_merged_gcps": s.get("n_merged_gcps"),
        "overlap_agreement": s.get("overlap_agreement"),
    }


def _json_value(v: str):
    """A `--l2-<field>` value: JSON when it parses (numbers, true/false, null, lists), else the string."""
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def l2_config(args) -> L2Config:
    """The preset `args.l2` with every given `--l2-<field>` applied."""
    over = {f.name: getattr(args, f"l2_{f.name}") for f in fields(L2Config) if hasattr(args, f"l2_{f.name}")}
    return L2Config.preset(args.l2, **over)


def l1_config(args) -> L1Config:
    """The preset `args.l1` with every given `--l1-<field>`, then `--fix-frame-defects`/`--fill-frame-defects`."""
    over = {f.name: getattr(args, f"l1_{f.name}") for f in fields(L1Config) if hasattr(args, f"l1_{f.name}")}
    over |= {k: v for k in ("fix_frame_defects", "fill_frame_defects") if (v := getattr(args, k)) is not None}
    return L1Config.preset(args.l1, **over)


def add_config_args(ap) -> None:
    """`--fix-frame-defects`/`--fill-frame-defects`, `--l1` and `--l2` presets and one `--l1-<field>`/`--l2-<field>`
    override per config field."""
    for step, h in (
        ("fix", "move each 4-band group onto true frames"),
        ("fill", "rebuild the lost frames (implies fix)"),
    ):
        ap.add_argument(
            f"--{step}-frame-defects",
            nargs="?",
            const=True,
            type=_json_value,
            default=None,
            help=f"{h}; unset keeps the L1 preset (iirspy L1: on, issdc-nci: off)",
        )
    for level, cls in (("l1", L1Config), ("l2", L2Config)):
        ap.add_argument(f"--{level}", default="iirspy", choices=list(cls.PRESETS), help=f"{cls.__name__} preset")
        for f in fields(cls):
            ap.add_argument(
                f"--{level}-{f.name.replace('_', '-')}",
                dest=f"{level}_{f.name}",
                type=_json_value,
                default=argparse.SUPPRESS,  # absent, not None: `--l2-polish null` is a real override
                help=f"override {cls.__name__}.{f.name}",
            )


def write_refl_outputs(
    job: Job,
    args,
    *,
    l1,
    l2,
    geo,
    ftif: Path,
    fl2: Path,
    fgcps: Path | None,
    gcps,
    cfg,
    lat_range,
    bands,
    scan0: int,
    gfiles: dict,
    sun: tuple,
    kernels: list,
    fgeom: Path,
    flabel: Path,
    prov_kw: dict,
    t_start: float,
) -> None:
    """L2 sanity stats, the summary and provenance json, tags on the L1/L2 rasters, and the `--keep`/`--keep-l1` copies."""
    from iirspy.iirs import L2
    from iirspy.l2 import stats as l2_stats

    work, sid, group = job.work, job.sid, job.group
    nci = args.l2 == "issdc-nci"
    l2cfg = args.l2_config
    ftopo = geo.sidecars.get("topo")
    az, elev, r_sun = sun
    t_stats = time.time()
    try:
        stats = l2_stats.l2_stats(
            fl2,
            ftif,
            np.asarray(l2.img.solar_inc),
            geo.loc.lat.values,
            sdist=float(l2.img.attrs["solar_distance_au"]),
        )
    except Exception as e:  # sanity stats must never fail a refl run
        stats = {"error": f"{type(e).__name__}: {e}"}
    log.info(f"l2_stats: {time.time() - t_stats:.1f}s {stats.get('error', 'ok')}")
    fglt = fgcps.parent / f"{sid}_{group}_glt.tif" if fgcps else None
    if fglt is not None:
        log.info(
            f"glt: {fglt}" if fglt.is_file() else f"glt: {fglt} not found -- iirs-georef-solve should have built it"
        )
    resources = f"{resource_report(l1.img)}; ncpu={ncpu()}"
    log.info(resources)

    notes = json.loads(l1.img.attrs["empirical_notes"]) if "empirical_notes" in l1.img.attrs else None
    topo_tags = {}
    if ftopo:
        with rasterio.open(ftopo) as src:
            topo_tags = {
                "path": str(ftopo),
                "bands_used": src.tags().get("bands_used"),
                "n_pieces": src.tags().get("n_pieces"),
            }
    summary = {
        "sid": sid,
        "group": group,
        "resources": resources,
        "georef": geo.method,
        "geometry_quality": geo.quality,
        "gcps": str(fgcps) if fgcps else None,
        "n_gcps": len(gcps) if gcps else None,
        "solve_quality": solve_quality(fgcps) if fgcps else None,
        "lat_range": list(lat_range),
        "bands": bands,
        "l2_preset": args.l2,
        "l1_source": "issdc-nci" if nci else "iirspy",
        "l2_config": l2cfg.to_attrs(),
        "topo": topo_tags,
        "aoi_m": list(cfg.aoi) if cfg else None,
        "scan0": scan0,
        "l1": ftif.name,
        "l2": fl2.name,
        "obs": gfiles["obs"].name,
        "obs_sensor_source": geo.tags["SENSOR_SOURCE"],
        "geometry": {k: f.name for k, f in gfiles.items()},
        "glt": str(fglt) if fglt else None,
        "sun_az_grid_deg": az,
        "sun_elev_deg": elev,
        "sun_angular_radius_deg": r_sun,
        "l1_empirical_notes": notes,
        "l2_diagnostics": {k: l2.img.attrs.get(k) for k in L2.DIAGNOSTICS},
        "l2_stats": stats,
        "total_s": round(time.time() - t_start, 1),
    }
    fsummary = work / f"{sid}_{group}_refl_summary.json"
    fsummary.write_text(json.dumps(summary, indent=1, default=str))
    log.info(f"\ndone in {summary['total_s']}s.")

    prov_inputs = {
        "nri_zip": file_info(job.zip),
        "gcps": file_info(fgcps, sha=True) if fgcps else None,
        "frame_defects": {**file_info(job.frame_defects), "key": defects_key(job.frame_defects)}
        if job.frame_defects
        else None,
        "label": file_info(flabel),
        "geometry_csv": file_info(fgeom),
        "geometry": geo.provenance,
        "spice_kernels": [str(k) for k in kernels],
        "topo": file_info(ftopo) if ftopo else None,
    }
    prov = io.provenance(
        **prov_kw,
        l2_preset=args.l2,
        l2_config=l2cfg.to_attrs(),
        inputs=prov_inputs,
        empirical_notes=notes,
        steps=json.loads(l2.img.attrs["provenance"]),
    )
    fprov = work / f"{sid}_{group}_provenance.json"
    fprov.write_text(json.dumps(prov, indent=1, default=str))
    for f in (ftif, fl2):
        with rasterio.open(f, "r+") as dst:
            dst.update_tags(IIRS_PROVENANCE=json.dumps(prov, default=str))

    sidecars = [
        f for f in (ftif.with_name(ftif.stem + "_empirical.npz"), ftif.with_name(ftif.stem + "_snr.tif")) if f.exists()
    ]
    vrt = lambda f: [f.with_suffix(".vrt")] if f.with_suffix(".vrt").exists() else []
    keep(work, args.keep_l1, ftif, *vrt(ftif), *sidecars, fprov)
    keep(work, args.keep, fl2, *vrt(fl2), *gfiles.values(), fsummary, fprov, job.logfile)
    if ftopo and fgcps:  # geometry, so it lands beside the GCPs rather than with L2
        keep(work, str(fgcps.parent) if args.keep else None, ftopo)
