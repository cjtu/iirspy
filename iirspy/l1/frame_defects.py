"""Frame defects in the raw IIRS cube: detect them once per strip, then re-time and fill radiance.

The detector reads out in 4-band groups (group g = bands 4g+1..4g+4, 1-based) and each group can lose frames on its
own, with nothing in the DN to say so:

- drop: from an exact line on, the group's raw row y holds frame y + 1. Lags only accumulate. A group can also start
  lagged (frames lost before the first measurable line).
- overwrite: frames y, y+1 are lost and refilled A A B B (row y repeats y-1, row y+1 repeats y+2); timing is kept.
- black: four all-zero rows in one group while the rest of the detector has signal; timing is kept.

`build_table` finds all three from one pass over the raw cube, every group measured against the one reference group
`REF`, and writes them to `<sid>_frame_defects.csv`.
`apply` moves every group onto true frames and masks the lost ones; `fill` rebuilds those from the pixel's own
spectrum. ISSDC's nci carries the same defects uncorrected. `fill_gaps` is the last L1 step: it calls `fill`, then
bridges the short NaN runs left by masking (up to `BAND_GAP` bands, one column across x, `ROW_GAP` rows along y).
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import warnings
import zipfile
from itertools import pairwise
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

from iirspy import io

NG = 64
K = 8  # max |offset| searched, lines (20210723T2153218013 reaches +6)
X0, X1, XBIN = 15, 235, 2  # cross-track span and binning of the matching rows
HPX = 15  # cross-track high-pass window, px
WIN = 1000  # pre-screen window, lines
STRIDE = 4  # pre-screen uses every STRIDE-th line
MARGIN = 0.01  # min per-line soft margin to accept a level
PEAK = 0.05  # min per-line correlation at the best level: below it the rows are noise
REF = 13  # bands 53-56, the reference every group is measured against
SKIP = {0, 1, 63}  # bands 1-8, 253-256: no usable signal
MAX_BRIDGE = 9  # widest band span `fill` bridges: a hole of two neighbouring groups
BAND_GAP = 9  # widest spectral hole `interpolate_na` bridges: two dropped groups (8 bands) in a row
ROW_GAP = 5  # likewise along y: up to 4 lost frames in a row


def retime(rad, qub, frame_defects=None):
    """Move each 4-band group of `rad` onto true frames; the lost frames ride along as the `lost_frames` coord.

    `frame_defects` is the strip's table (`build_table`). None builds it from the full cube `qub` and caches it beside
    it as `<qub stem>_frame_defects.csv`; a band-subset cube cannot be scanned, so `rad` comes back unchanged, with a
    warning.
    """
    if frame_defects is None:
        frame_defects = Path(qub).with_name(f"{Path(qub).stem}_frame_defects.csv")
        if not frame_defects.exists():
            try:
                build_table(qub, frame_defects)
            except ValueError as e:
                warnings.warn(f"fix_frame_defects skipped: {e}", UserWarning, stacklevel=2)
                return rad
    out, lost = apply(rad, frame_defects)
    return out.assign_coords(lost_frames=(("band", "y"), lost))


#  reading


def _bands(path):
    """(nl, ns, iterator of (band index, image)) for an nri zip (streamed) or an ENVI .qub (memmapped)."""
    path = Path(path)
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        names = {Path(n).suffix: n for n in z.namelist() if not n.endswith("/")}
        hdr = z.read(names[".hdr"]).decode()
    else:
        z, hdr = None, path.with_suffix(".hdr").read_text()
    ns, nl, nb, dt = (
        int(re.search(rf"^{k}\s*=\s*(\d+)", hdr, re.M)[1])  # type: ignore[index]
        for k in ("samples", "lines", "bands", "data type")
    )
    dtype = np.dtype({2: "<i2", 4: "<f4"}[dt])
    if nb != 4 * NG:
        raise ValueError(f"{path.name} holds {nb} bands; the frame-defect table needs all {4 * NG}")

    def it():
        if z is None:
            mm = np.memmap(path, dtype, "r", shape=(nb, nl, ns))
            for b in range(nb):
                yield b, np.asarray(mm[b])
        else:
            with z.open(names[".qub"]) as f:
                for b in range(nb):
                    yield b, np.frombuffer(f.read(nl * ns * dtype.itemsize), dtype).reshape(nl, ns)

    return nl, ns, it()


def _lit(prof):
    """Lines where the detector has signal (median group profile above 5% of its strip median)."""
    others = np.median(prof[[g for g in range(NG) if g not in SKIP]], axis=0)
    return others > 0.05 * np.median(others)


def scan(path):
    """One pass over the cube: per-group normalised high-passed rows, profiles, and copied / all-zero row flags."""
    nl, ns, it = _bands(path)
    nx = (X1 - X0) // XBIN
    rows = np.zeros((NG, nl, nx), np.float16)
    prof = np.zeros((NG, nl), np.float32)
    dup = np.ones((NG, nl), bool)
    zero = np.ones((NG, nl), bool)
    acc = np.zeros((nl, ns), np.float32)
    for b, im in it:
        g = b // 4
        dup[g, 0] = False
        # copies are exact or off by 1 DN in a band, never more; noise is tens of DN
        dup[g, 1:] &= np.abs(np.diff(im[:, X0:X1].astype(np.int32), axis=0)).max(1) <= 1
        zero[g] &= (im == 0).all(1)
        acc += im
        if b % 4 == 3:
            a = acc[:, X0:X1].reshape(nl, nx, XBIN).mean(2) / 4
            prof[g] = a.mean(1)
            h = a - uniform_filter1d(a, max(1, HPX // XBIN), axis=1)
            h -= h.mean(1, keepdims=True)
            rows[g] = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-6)
            acc[:] = 0
    rows[:, ~_lit(prof)] = 0  # unlit lines carry noise, not texture: they take no part in matching
    return rows, prof, dup, zero


#  matching


def _corr(rows, g, ref, y0, y1, stride=1):
    """(n, 2K+1) correlation of group g's lines y0:y1:stride with ref rows y+k; -1 where ref is out of range."""
    nl = rows.shape[1]
    y = np.arange(max(y0, 0), min(y1, nl), stride)
    a = rows[g, y].astype(np.float32)
    corr = np.full((len(y), 2 * K + 1), -1.0, np.float32)
    for j, k in enumerate(range(-K, K + 1)):
        ok = (y + k >= 0) & (y + k < nl)
        corr[ok, j] = (a[ok] * rows[ref, y[ok] + k].astype(np.float32)).sum(1)
    return corr


def _soft_level(corr):
    """Offset maximising the summed correlation, and its margin over the runner-up per line."""
    s = corr.sum(0)
    o = np.argsort(s)[::-1]
    n = max(len(corr), 1)
    return int(o[0] - K), float((s[o[0]] - s[o[1]]) / n) if s[o[0]] / n > PEAK else 0.0


def _step_scores(corr, a, b):
    """Score of a step from level a to level b at each line: summed correlation of a before it and b after."""
    ca, cb = corr[:, a + K], corr[:, b + K]
    return np.r_[0, np.cumsum(ca)] + np.r_[np.cumsum(cb[::-1])[::-1], 0]


def detect(rows, tol=3):  # noqa: C901
    """Per-group events [(line, group, delta)]; line 0 holds a group's start lag.

    Outward from `REF`, each group is compared with the nearest of its (up to three) neighbours toward REF, per WIN
    window, and the neighbours vote on its offset. Only the level against that anchor is localised to a line
    (adjacent groups correlate strongly and barely drift), and a group inherits its anchor's events: lag = anchor
    lag + level. A -1 against REF is a drop of REF, which the neighbour's own +1 cancels; other -1 are noise. A group
    decided in under half the windows takes its anchor's events. If starts come out negative, all groups are bumped by the most
    negative one shared by two groups: the lost frames sit in the unmeasurable start of the strip.
    """
    nl, w = rows.shape[1], WIN // STRIDE
    live = [g for g in range(NG) if g not in SKIP]
    off = np.full((NG, nl // STRIDE // w), np.nan)
    off[REF] = 0
    ev: dict[int, list[tuple[int, int]]] = {REF: []}
    decided = {REF}  # groups that measured something; the others copy their anchor
    for side in (1, -1):
        for g in sorted((g for g in live if (g - REF) * side > 0), key=lambda g: abs(g - REF)):
            anchors = sorted(
                (a for a in decided if 0 <= (a - REF) * side < (g - REF) * side), key=lambda a: abs(a - g)
            )[:3]
            est = np.full((len(anchors), len(off[g])), np.nan)
            for j, a in enumerate(anchors):
                c = _corr(rows, g, a, 0, nl, STRIDE)
                for i in np.flatnonzero(~np.isnan(off[a])):
                    lv, m = _soft_level(c[i * w : (i + 1) * w])
                    if m > MARGIN:
                        est[j, i] = off[a, i] + lv
            off[g] = np.round(np.ma.median(np.ma.masked_invalid(est), axis=0).filled(np.nan))
            good = np.flatnonzero(~np.isnan(off[g]))
            v = off[g, good]
            off[g, good[1:-1][(v[1:-1] != v[:-2]) & (v[:-2] == v[2:])]] = np.nan  # a lone window off its neighbours
            a0 = anchors[0]
            if np.isnan(off[g]).sum() > 0.5 * len(off[g]):  # decided in under half the windows: no evidence
                off[g] = off[a0]
            else:
                decided.add(g)
            lvl = off[g] - off[a0]
            good = np.flatnonzero(~np.isnan(lvl))
            own = []
            if len(good):
                own = [(0, int(lvl[good[0]]), 0, 0)] if lvl[good[0]] else []
                for i in np.flatnonzero(np.diff(lvl[good])):
                    y0, y1 = good[i] * WIN, (good[i + 1] + 1) * WIN
                    a, b = (int(np.clip(lvl[good[j]], -K, K)) for j in (i, i + 1))
                    c = _corr(rows, g, a0, y0, y1)
                    step, cur = (1 if b > a else -1), 0
                    for t in range(a + step, b + step, step):
                        k = int(np.argmax(_step_scores(c[cur:], t - step, t)))
                        own.append((y0 + cur + k, step, y0, y1))
                        cur += k
            if a0 == REF:  # a step down against REF is a drop of REF
                ev[REF] += [(y, 1) for y, d, *_ in own if d < 0 and not any(abs(y - r) <= tol for r, _ in ev[REF])]
            ev[g] = list(ev[a0])
            for y, d, y0, y1 in own:
                inside = [e for e in ev[g] if d < 0 < e[1] and e[0] > 0 and y0 - tol <= e[0] < y1 + tol]
                if inside:  # the anchor's drop this group did not share
                    ev[g].remove(min(inside, key=lambda e: abs(e[0] - y)))
                elif d > 0 or y == 0:
                    ev[g].append((y, d))
    start = {g: sum(d for y, d in ev[g] if y == 0) for g in live}
    sv = sorted(start[g] for g in decided)
    bump = -min(next((v for v, w in pairwise(sv) if v == w), 0), 0)
    out = [(0, g, start[g] + bump) for g in live if start[g] + bump]
    out += [(y, g, d) for g in live for y, d in ev[g] if y > 0]
    return sorted(out, key=lambda e: (e[0], e[1]))


# ---------------------------------------------------------------- table


def _runs(flag):
    """[(start, length)] of True runs."""
    d = np.diff(np.r_[0, flag.astype(int), 0])
    return [(int(a), int(b - a)) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True)]


def _fills(dup, zero, prof):
    """(kind, line, group, rows) for black and overwrite fills, only where the rest of the detector has signal."""
    lit = _lit(prof)
    out = []
    for g in sorted(set(range(NG)) - SKIP):
        out += [("black", s, g, n) for s, n in _runs(zero[g] & lit)]
        d = dup[g] & ~zero[g] & lit
        partner = set()
        for y in np.flatnonzero(d):
            if y in partner:
                continue
            if y + 2 < len(d) and d[y + 2]:
                out.append(("overwrite", int(y), g, 2))
                partner.add(y + 2)
            else:  # ponytail: a lone copy masks both A and B readings; match against the reference to save 2 rows
                out.append(("overwrite", max(int(y) - 2, 0), g, 4))
    return out


def build_table(src, fcsv) -> Path:
    """Detect frame defects in a raw nri zip or .qub and write them to `fcsv`.

    Columns kind,line,group,bands,value: `start`/`drop` give the lag change at that raw line (old tables may hold value None: start
    undetermined, taken as 0); `overwrite`/`black` give the number of raw rows from that line that hold no frame.
    The write is atomic, so tasks racing on one strip never read a partial table.
    """
    rows, prof, dup, zero = scan(src)
    events = detect(rows)
    lines = ["kind,line,group,bands,value"]
    lines += [f"{'start' if y == 0 else 'drop'},{y},{g},{4 * g + 1}-{4 * g + 4},{d}" for y, g, d in events]
    lines += [f"{k},{y},{g},{4 * g + 1}-{4 * g + 4},{n}" for k, y, g, n in _fills(dup, zero, prof)]
    out = Path(fcsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, out)
    return out


def db_name(raw) -> str:
    """Member name of raw zip or cube `raw`'s table in a frame-defect db: `<sid>_<data center>_frame_defects.csv`.

    Each data center assembles its own L0, so `start`/`black`/`overwrite` rows differ between copies of a strip (S114).

    >>> db_name("ch2_iir_nri_20230604T1425413885_d_img_n18.zip")
    '20230604T1425413885_n18_frame_defects.csv'
    """
    return f"{io.iirsbasename(Path(raw).name)}_{io.data_center(raw)}_frame_defects.csv"


def from_db(db, raw, fcsv) -> bool:
    """Copy raw zip `raw`'s table out of the zip `db` (members named by `db_name`) to `fcsv`; False if it has none.

    `fcsv` takes the member's date as mtime, the table's build date, so GCPs solved before it read as solved on
    un-retimed cubes (`predates_frame_defects`).
    """
    if not Path(db).exists():
        return False
    with zipfile.ZipFile(db) as z:
        try:
            info = z.getinfo(db_name(raw))
        except KeyError:
            return False
        tmp = Path(fcsv).with_name(f"{Path(fcsv).name}.{os.getpid()}.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(z.read(info))
    t = time.mktime((*info.date_time, 0, 0, -1))
    os.utime(tmp, (t, t))
    os.replace(tmp, fcsv)
    return True


def table_key(fcsv) -> str:
    """Content hash of a frame-defect table, so a cache built on another (or no) table never matches."""
    return hashlib.sha256(Path(fcsv).read_bytes()).hexdigest()[:12]


def read_table(fcsv):
    """(events [(line, group, delta)], masked [(line, group, rows)]) from a `build_table` csv."""
    events, masked = [], []
    for ln in Path(fcsv).read_text().splitlines()[1:]:
        kind, y, g, _, v = ln.split(",")
        if kind in ("start", "drop"):
            events.append((int(y), int(g), None if v == "None" else int(v)))
        else:
            masked.append((int(y), int(g), int(v)))
    return events, masked


def lag_table(events, nl):
    """(NG, nl) lag of each group at each raw row: raw row y holds frame y + lag.

    >>> lag_table([(0, 3, 1), (5, 3, 1), (0, 4, None)], 8)[3:5].tolist()
    [[1, 1, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]]
    """
    lag = np.zeros((NG, nl), np.int16)
    for y, g, d in events:
        if d is not None:
            lag[g, y:] += d
    return lag


def sources(fcsv, nl):
    """(NG, nl) raw row holding each true frame of each group, -1 where the frame was lost.

    >>> import tempfile
    >>> f = Path(tempfile.mkdtemp()) / "t.csv"
    >>> _ = f.write_text("kind,line,group,bands,value\\ndrop,3,0,1-4,1\\noverwrite,6,1,5-8,2\\n")
    >>> src = sources(f, 9)
    >>> src[0].tolist(), src[1].tolist()
    ([0, 1, 2, -1, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, -1, -1, 8])
    """
    events, masked = read_table(fcsv)
    lag = lag_table(events, nl)
    src = np.full((NG, nl), -1, np.int64)
    raw = np.arange(nl)
    for g in range(NG):
        f = raw + lag[g]
        ok = (f >= 0) & (f < nl)
        src[g, f[ok]] = raw[ok]
    for y, g, n in masked:
        src[g, np.isin(src[g], np.arange(y, y + n))] = -1
    return src


# radiance


def apply(rad, fcsv):
    """`rad` (band, y, x) with every group moved onto true frames, and the (band, y) mask of lost frames.

    `rad` is any y-range of the full strip the table was built on: its y coords are raw line numbers. A lost frame
    (dropped, copied or black, or sourced from outside `rad`) becomes NaN.
    """
    lines = np.floor(rad.y.values).astype(np.int64)
    src = sources(fcsv, int(lines[-1]) + 1)[:, lines] - lines[0]
    src[(src < 0) | (src >= len(lines))] = -1
    groups = (rad.band.values - 1) // 4
    lost = src[groups] < 0
    rad = rad.transpose("band", "y", "x")
    data = rad.data if isinstance(rad.data, da.Array) else da.from_array(rad.data, chunks=(1, -1, -1))
    same = np.arange(len(lines))
    pieces = [
        data[i : i + 1] if (src[g] == same).all() else data[i : i + 1, np.maximum(src[g], 0)]
        for i, g in enumerate(groups)
    ]
    out = rad.copy(data=da.concatenate(pieces, axis=0))
    return out.where(~xr.DataArray(lost, dims=("band", "y"))), lost


def fill(rad, lost, depth=6):
    """Rebuild lost frames from the pixel's own spectrum, shaped by the rows above and below.

    Each run of lost bands is bridged linearly between the nearest kept bands either side, at that pixel; the bridge
    is then scaled by how far the nearest kept rows (within `depth`) depart from their own bridge, which carries the
    absorption the straight line misses. Faked holes in good data come back at the band noise floor; a linear bridge
    alone or along-track interpolation miss by 1.5-3x that. Runs wider than MAX_BRIDGE bands or without a kept
    neighbour stay NaN.

    A curved spectrum on a varying field, bands 5-8 dropping a frame at line 3, comes back exactly:

    >>> import tempfile, xarray as xr
    >>> f = Path(tempfile.mkdtemp()) / "t.csv"
    >>> _ = f.write_text("kind,line,group,bands,value\\ndrop,3,1,5-8,1\\n")
    >>> s = 1 + 0.02 * (np.arange(1, 13) - 6.0) ** 2
    >>> truth = s[:, None, None] * np.linspace(1, 2, 24).reshape(1, 8, 3)
    >>> raw = truth.copy()
    >>> raw[4:8, 3:-1] = truth[4:8, 4:]
    >>> rad = xr.DataArray(raw, dims=("band", "y", "x"), coords={"band": np.arange(1, 13), "y": np.arange(8)})
    >>> fixed, lost = apply(rad, f)
    >>> np.flatnonzero(lost[4]).tolist()
    [3]
    >>> bool(np.allclose(fill(fixed, lost).values, truth))
    True
    """
    nums = rad.band.values
    rad = rad.transpose("band", "y", "x")
    data = rad.data if isinstance(rad.data, da.Array) else da.from_array(rad.data)
    data = data.rechunk({0: -1, 2: -1})
    mask = da.from_array(lost[:, :, None], chunks=(-1, data.chunks[1], 1))

    def block(a, m):
        a, m = a.copy(), m[:, :, 0]
        for y in np.flatnonzero(m.any(0)):
            for b0, n in _runs(m[:, y]):
                lo, hi = b0 - 1, b0 + n
                if lo < 0 or hi >= len(nums) or nums[hi] - nums[lo] > MAX_BRIDGE:
                    continue
                t = ((nums[b0:hi] - nums[lo]) / (nums[hi] - nums[lo]))[:, None]
                kept = ~m[lo : hi + 1].any(0)
                up = [r for r in range(y - 1, max(y - depth, -1), -1) if kept[r]][:1]
                down = [r for r in range(y + 1, min(y + depth, m.shape[1])) if kept[r]][:1]
                if not up + down:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixel: shape falls back to 1
                    shape = np.nanmean([a[b0:hi, r] / (a[lo, r] * (1 - t) + a[hi, r] * t) for r in up + down], axis=0)
                a[b0:hi, y] = (a[lo, y] * (1 - t) + a[hi, y] * t) * np.clip(np.nan_to_num(shape, nan=1.0), 0.5, 2)
        return a

    filled = da.map_overlap(block, data, mask, depth={0: 0, 1: depth, 2: 0}, boundary="none", dtype=data.dtype)
    return rad.copy(data=filled)


def fill_gaps(rad, *, fill_frame_defects=False, interp_bands=None, interp_spatial=False):
    """`rad` with lost frames rebuilt (`fill_frame_defects`), then band gaps (`interp_bands` method) and x/y gaps
    filled.

    The `lost_frames` coord is consumed.
    """
    lost = rad.lost_frames.values if "lost_frames" in rad.coords else None
    out = rad.drop_vars("lost_frames", errors="ignore")
    if fill_frame_defects and lost is not None:
        out = fill(out, lost)
    if interp_bands is not None:
        out = out.interpolate_na("band", max_gap=BAND_GAP, keep_attrs=True, method=interp_bands)
    if interp_spatial:
        # max_gap=2 in 0.5-based x coords spans exactly one missing column
        out = out.interpolate_na("x", max_gap=2, keep_attrs=True, method="linear")
        out = _interp_rows(out)
    return out


def _interp_rows(rad, max_gap=ROW_GAP):
    """Linear fill along y of NaN runs shorter than `max_gap` rows, chunk by chunk (overlapped so a run on a chunk edge fills).

    A 2-row run straddling the chunk edge fills; a run of `max_gap` rows or more stays NaN:

    >>> a = np.ones((1, 12, 1)); a[0, 5:7] = np.nan; a[0, 8:] = np.nan
    >>> r = xr.DataArray(da.from_array(a, chunks=(1, 6, 1)), dims=("band", "y", "x"))
    >>> np.isnan(_interp_rows(r, 4).values[0, :, 0]).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    ax = rad.get_axis_num("y")
    data = rad.data if isinstance(rad.data, da.Array) else da.from_array(rad.data)
    data = data.rechunk({a: -1 for a in range(rad.ndim) if a != ax})

    def block(b):
        dims = [f"d{i}" for i in range(b.ndim)]
        dims[ax] = "y"
        return (
            xr.DataArray(b, dims=dims, coords={"y": np.arange(b.shape[ax])}).interpolate_na("y", max_gap=max_gap).values
        )

    return rad.copy(data=data.map_overlap(block, depth={ax: max_gap}, boundary="none", dtype=data.dtype))
