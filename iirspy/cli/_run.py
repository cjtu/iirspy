from __future__ import annotations

import datetime
import errno
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from iirspy import instrument, io
from iirspy.georef import ancillary
from iirspy.georef import chunks as ck
from iirspy.l1.config import CALIB
from iirspy.l1.frames import table_key as frames_key
from iirspy.workspace import Workspace, ncpu, phase


def paths_from_env() -> io.Paths:
    archive = Path(os.environ.get("IIRS_ARCHIVE", "/mnt/d/ddata/moon/ch2_iirs"))
    return io.Paths(
        archive=archive,
        anc_roots=tuple(Path(p) for p in os.environ["IIRS_ANC_ROOTS"].split(":"))
        if "IIRS_ANC_ROOTS" in os.environ
        else (archive,),
        dem_roots=tuple(
            Path(p)
            for p in os.environ.get("IIRS_DEM_ROOTS", "/mnt/d/ddata/moon/lro_lola:/mnt/d/ddata/moon/lro_wac").split(":")
        ),
        spice=Path(os.environ.get("IIRS_SPICE", str(archive / "spice"))),
        recal_root=Path(os.environ.get("IIRS_RECAL_ROOT", str(Path.home() / "data" / "ch2_iirs"))),
    )


@dataclass(frozen=True)
class Job:
    sid: str
    group: str
    zip: Path
    stage: Path
    logfile: Path
    paths: io.Paths
    ws: Workspace
    frames: Path | None = None


def open_run(sid: str, group: str, out: str | None, step: str, paths: io.Paths, deadline: float | None = None) -> Job:
    """Locate the scene's zip and prepare the work dir, `<sid>_<group>_<step>.log` and stage."""
    if not paths.archive.exists():
        sys.exit(f"{paths.archive} missing -- archive not mounted, set IIRS_ARCHIVE")
    zip_path = ck.nri_zip(sid, paths)
    if zip_path is None:
        sys.exit(f"no nri zip for {sid} under {paths.archive / 'zips'}")
    work = Path(out) if out else Path.cwd() / "runs" / f"{sid}_{group}"
    logfile = work / f"{sid}_{group}_{step}.log"
    work.mkdir(parents=True, exist_ok=True)
    if os.environ.get("IIRS_STAGE"):
        stage = Path(os.environ["IIRS_STAGE"])
    elif os.environ.get("SLURM_TMPDIR"):
        stage = Path(os.environ["SLURM_TMPDIR"]) / "stage"
    else:
        stage = work.parent / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    logfile.write_text("")

    def log(msg):
        print(msg, flush=True)
        with logfile.open("a") as f:
            f.write(str(msg) + "\n")

    return Job(sid, group, zip_path, stage, logfile, paths, Workspace(work, deadline, log))


def log_provenance(ws: Workspace) -> None:
    """Slurm allocation + host details, `seff`-style, so a slow/failed run can be triaged later.

    Env vars are unset outside Slurm (e.g. a dev-box run), so every lookup falls back to "n/a" or
    the plain hostname/cpu count instead of raising.
    """
    import getpass
    import platform

    def _gb(mb: str) -> str:
        try:
            return f"{int(mb) / 1024:.0f}G"
        except ValueError:
            return "n/a"

    log = ws.log
    job_id = os.environ.get("SLURM_JOB_ID")
    array_job, array_task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    log("--- job ---")
    log(f"Job ID: {job_id or 'n/a'}")
    if array_job:
        log(f"Array Job ID: {array_job}_{array_task}")
    log(f"Cluster: {os.environ.get('SLURM_CLUSTER_NAME', 'n/a')}")
    log(f"User: {getpass.getuser()}")
    log(f"Node: {os.environ.get('SLURMD_NODENAME', platform.node())}")
    log(f"Cores per node: {os.environ.get('SLURM_CPUS_PER_TASK', ncpu())} (ncpu()={ncpu()})")
    log(f"Memory: {_gb(os.environ.get('SLURM_MEM_PER_NODE', ''))}/node")
    log(f"Python: {platform.python_version()}")


def stage_inputs(job: Job, day: str, bands: list[int]) -> Path:
    """Extract the zip and assemble the one-root PDS tree `L0` reads. Returns the raw cube's path.

    The zip brings every nri member (cube bands, spm, oat, xml); only the nci products it lacks are
    copied in. The geometry csv comes from every `paths.anc_roots`, not just the archive: without it
    `l1.csv` is None and the crop dies in `parse_geom` after the whole cube has calibrated.
    """
    raw_qub = job.stage / f"data/raw/{day}/{job.zip.stem}.qub"
    if not raw_qub.exists():
        job.ws.log(f"extracting {job.zip.name} ({len(bands)} bands) -> {job.stage}")
        with phase(job.ws, "extraction"):
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


CALIBRATE_KWS = {"empirical": True, "interp_bands": "linear", "interp_spatial": True, "bad_pixel_mask": True}


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
    its full resolved fields, plus a G1 table's content hash, are the cache key.

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
    effective_calibrate_kws = {**CALIBRATE_KWS, "frame_offsets": str(job.frames), **(calibrate_kwargs or {})}
    if l1_config is not None:
        effective_calibrate_kws = {"l1_config": l1_config.to_attrs(), "frame_offsets": str(job.frames)}
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
            and meta.get("frames") == frames_key(job.frames)
            and meta.get("crop_geom") == crop_geom
        ):
            job.ws.log(f"L1 already built: {ftif_out} (covers {cached_range}, requested {lat_range})")
            return ftif_out, meta["scan0"], cached_range
        job.ws.log(
            f"cached L1 {cached_range}/bands={meta.get('bands')}/"
            f"calibrate_kwargs={meta.get('calibrate_kwargs', '<unrecorded>')} does not match requested "
            f"{lat_range}/{bands}/{effective_calibrate_kws} -- rebuilding"
        )
    from iirspy import L0

    with phase(job.ws, "staging"):
        raw_qub = stage_inputs(job, day, bands)

    # `chunk_y` sets both the dask y-chunk and, through `_row_block`, the write block.
    with phase(job.ws, "L0 read + calibrate (graph build; the compute lands in the write)"):
        l0 = L0(job.sid, job.stage, chunk={"band": -1, "y": chunk_y, "x": -1})
        l1 = (
            l0.calibrate(**effective_calibrate_kws)
            if l1_config is None
            else l0.calibrate(l1_config, frame_offsets=str(job.frames))
        )
    # l1.img already holds exactly `bands` -- that's what _stage_inputs staged.
    _, xyext = ancillary.parse_geom(fgeom or l1.csv, latlonextent=(-180, 180, *lat_range))
    ymin, ymax = xyext[2], xyext[3]
    job.ws.log(f"L1 crop lat={lat_range}: scan {ymin}-{ymax} ({ymax - ymin} rows)")
    l1.img = l1.img.sel(y=slice(ymin, ymax))
    l1.img = l1.img / instrument.RAD_NATIVE_SCALE
    l1.img.attrs["units"] = "1000 mW/cm^2/sr/um"
    if out_bands is not None:
        l1.img = l1.img.sel(band=out_bands)
    with phase(job.ws, f"L1 write ({l1.img.shape[0]} band x {l1.img.shape[1]} rows, chunk_y={chunk_y})"):
        l1.save(str(ftif_out))
    fmeta.write_text(
        json.dumps({
            "scan0": int(ymin),
            "lat_range": list(lat_range),
            "bands": saved_bands,
            "calibrate_kwargs": effective_calibrate_kws,
            "frames": frames_key(job.frames),
            "crop_geom": crop_geom,
        })
    )
    if not keep_raw:
        raw_qub.unlink(missing_ok=True)
        raw_qub.with_suffix(".hdr").unlink(missing_ok=True)
    job.ws.log(f"L1 saved: {ftif_out}")
    return ftif_out, int(ymin), lat_range


def build_l1_nci(
    job: Job,
    lat_range: tuple[float, float],
    bands: list[int],
    ftif_out: Path,
    chunk_y: int = 1024,
    fix_frames: bool = False,
    fill_frames: bool = False,
) -> tuple[Path, int, tuple[float, float]]:
    """`build_l1`'s counterpart for ISSDC's own L1: the nci cube, cropped to `lat_range`, written in the same
    layout (native radiance units, `bands` only, `.meta.json` beside it) so everything after it is shared.

    The cube is unzipped into the ISSDC tree under `paths.archive` if needed (`ck.ensure_nci`). No empirical dark/flat
    or band repair: ISSDC's calibration as shipped, plus `IIRSData.fix_frames` with `job.frames` if asked (`fill_frames`
    implies `fix_frames`). Returns the L1 path, row 0's `Scan` and `lat_range`.
    """
    from iirspy import L1

    fix_frames = fix_frames or fill_frames
    fmeta = ftif_out.with_suffix(".meta.json")
    if ftif_out.exists() and fmeta.exists():
        meta = json.loads(fmeta.read_text())
        if (
            meta.get("source") == "issdc-nci"
            and meta.get("bands") == bands
            and tuple(meta["lat_range"]) == tuple(lat_range)
            and meta.get("frames") == [fix_frames, fill_frames, frames_key(job.frames) if fix_frames else None]
        ):
            job.ws.log(f"nci L1 already built: {ftif_out}")
            return ftif_out, meta["scan0"], lat_range
    with phase(job.ws, "nci unzip"):
        ck.ensure_nci(job.sid, job.paths)
    with phase(job.ws, "nci L1 read + crop"):
        l1 = L1(
            job.sid, job.paths.archive, lonlatextent=(-180, 180, *lat_range), chunk={"band": -1, "y": chunk_y, "x": -1}
        )
        scan0 = int(l1.extent[2])
        l1.img = l1.img.sel(band=bands) / instrument.RAD_NATIVE_SCALE
        l1.img.attrs["units"] = "1000 mW/cm^2/sr/um"
        if fix_frames:
            l1.fix_frames(job.frames, fill_frames)
    with phase(job.ws, f"nci L1 write ({l1.img.shape[0]} band x {l1.img.shape[1]} rows)"):
        l1.save(str(ftif_out), snr_sidecar=False, empirical_sidecar=False)
    frames_meta = [fix_frames, fill_frames, frames_key(job.frames) if fix_frames else None]
    fmeta.write_text(
        json.dumps({
            "scan0": scan0,
            "lat_range": list(lat_range),
            "bands": bands,
            "source": "issdc-nci",
            "frames": frames_meta,
        })
    )
    job.ws.log(f"nci L1 saved: {ftif_out}")
    return ftif_out, scan0, lat_range


def frame_offsets_path(sid: str, paths: io.Paths, dest: Path | None = None) -> Path:
    """`sid`'s frame table: one per strip, beside its GCPs and `_L1_b*.tif` in `dest` (default `ck.recal_dir`)."""
    return Path(dest or ck.recal_dir(sid, paths)) / f"{sid}_frames_offsets.csv"


def ensure_frame_offsets(job: Job, dest: Path | None = None) -> Path:
    """`job.sid`'s frame table, built from the whole zip on first use and cached for every later task and group.

    It needs every band of every line once (~1 min); nothing else in a run does, so it streams the zip rather than
    staging it.
    """
    from iirspy.l1 import frames

    f = frame_offsets_path(job.sid, job.paths, dest)
    if not f.exists():
        job.ws.log(f"frame table missing, building from {job.zip.name} -> {f}")
        with phase(job.ws, "frame table"):
            frames.build_table(job.zip, f)
    return f


def predates_frames(fgcps: Path, fframes: Path) -> bool:
    """Whether `fgcps` was solved before `fframes` existed, i.e. on a cube whose groups were not re-timed."""
    return not fframes.exists() or fgcps.stat().st_mtime < fframes.stat().st_mtime


def keep(ws: Workspace, dest_dir: str | None, *files: Path) -> None:
    """Copy products out of the work dir into `dest_dir` (a no-op without one)."""
    if not dest_dir:
        return
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / f.name)
    ws.log(f"kept {len(files)} file(s) -> {dest}")


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
    """Whether the cube's y-blocks keep `ncpu` dask threads busy, and the peak RSS against the Slurm allocation."""
    import resource

    n, ny, nx = img.shape
    ychunks = img.chunks[img.dims.index("y")] if img.chunks is not None else (ny,)
    block_mb = n * max(ychunks) * nx * 4 / 2**20
    busy = "ok" if len(ychunks) >= ncpu() else "STARVED: fewer blocks than cores"
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    limit = os.environ.get("SLURM_MEM_PER_NODE")
    of = f" of --mem={int(limit) / 1024:.0f}G ({100 * peak_mb / int(limit):.0f}%)" if limit else ""
    return f"L1 {img.shape} in {len(ychunks)} y-blocks of <={block_mb:.0f} MB for {ncpu()} workers ({busy}); peak RSS {peak_mb:.0f} MB{of}"
