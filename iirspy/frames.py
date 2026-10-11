"""Frame defects in the raw IIRS cube: detect them once per strip, then re-time and fill radiance.

The detector reads out in 4-band groups (group g = bands 4g+1..4g+4, 1-based) and each group can lose frames on its
own, with nothing in the DN to say so:

- drop: from an exact line on, the group's raw row y holds frame y + 1. Lags only accumulate.
- overwrite: frames y, y+1 are lost and refilled A A B B (row y repeats y-1, row y+1 repeats y+2); timing is kept.
- black: four all-zero rows in one group while the rest of the detector has signal; timing is kept.

`build_table` finds all three from one pass over the raw cube and writes them to `<sid>_frames_offsets.csv`.
`apply` moves every group onto true frames and masks the lost ones; `fill` rebuilds those from the pixel's own
spectrum. ISSDC's nci carries the same defects uncorrected.
"""

from __future__ import annotations

import os
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

NG = 64
K = 5  # max |offset| searched, lines
X0, X1, XBIN = 15, 235, 2  # cross-track span and binning of the matching rows
HPX = 15  # cross-track high-pass window, px
WIN = 1000  # pre-screen window, lines
STRIDE = 4  # pre-screen uses every STRIDE-th line
MARGIN = 0.01  # min per-line soft margin to accept a level
REF_REFL = 29  # bands 117-120
REF_THERM = 50  # bands 201-204
THERMAL = range(43, 63)  # bands 173-252 sit in thermal emission, matched against REF_THERM
SKIP = {0, 1, 63}  # bands 1-8, 253-256: no usable signal
MAX_BRIDGE = 9  # widest band span `fill` bridges: a hole of two neighbouring groups


# ---------------------------------------------------------------- reading


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
        raise ValueError(f"{path.name} holds {nb} bands; the frame table needs all {4 * NG}")

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
    return rows, prof, dup, zero


# ---------------------------------------------------------------- matching


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
    return int(o[0] - K), float((s[o[0]] - s[o[1]]) / max(len(corr), 1))


def _window_levels(rows, g, ref):
    """Soft offset per WIN window from every STRIDE-th line, and whether each window is decided."""
    c = _corr(rows, g, ref, 0, rows.shape[1], STRIDE)
    w = WIN // STRIDE
    lv = [_soft_level(c[i : i + w]) for i in range(0, len(c) - w + 1, w)]
    return np.array([v[0] for v in lv], float), np.array([v[1] > MARGIN for v in lv])


def _soft_step(corr, a, b):
    """Index where level a before and level b after score highest together."""
    ca, cb = corr[:, a + K], corr[:, b + K]
    return int(np.argmax(np.r_[0, np.cumsum(ca)] + np.r_[np.cumsum(cb[::-1])[::-1], 0]))


def _levels(a, b):
    step = 1 if b > a else -1
    return list(range(a + step, b + step, step))


def detect(rows, prof, passes=3):
    """Per-group events [(line, group, delta)]; line 0 is the starting lag, delta None where it is undetermined.

    Re-runs on the re-timed rows until no residual step is left.
    """
    nl = rows.shape[1]
    ev, _ = _detect_once(rows)
    for _ in range(passes - 1):
        lag = lag_table(ev, nl)
        fr = np.zeros_like(rows)
        for g in range(NG):
            f = np.arange(nl) + lag[g]
            ok = (f >= 0) & (f < nl)
            fr[g, f[ok]] = rows[g, ok]
        new, _ = _detect_once(fr, starts_known=True)
        new = [(int(y - lag[g, min(y, nl - 1)]), g, d) for y, g, d in new if y > 0 and d]
        if not new:
            break
        ev = sorted(ev + new, key=lambda e: (e[0], e[1]))
    return ev


def _detect_once(rows, starts_known=False):
    """Events of each group relative to its block's reference, assembled to absolute lags (see `_assemble`).

    A stride-STRIDE pre-screen finds the windows where a group's offset changes; only those are matched line by line.
    """
    nl = rows.shape[1]
    groups = [g for g in range(NG) if g not in SKIP]
    refs = {g: (REF_THERM if g in THERMAL else REF_REFL) for g in groups if g not in (REF_REFL, REF_THERM)}
    rel, starts = {}, {}
    for g, ref in refs.items():
        level = 0 if starts_known else None
        for s in [] if starts_known else range(0, nl - WIN // 2, WIN // 2):
            lv, m = _soft_level(_corr(rows, g, ref, s, s + WIN))
            if m > MARGIN:
                level = lv
                break
        starts[g] = level
        if level is None:
            continue
        lv_w, ok = _window_levels(rows, g, ref)
        good = np.where(ok)[0]
        ev = []
        for i in np.where(np.diff(lv_w[good]) != 0)[0]:  # each change between consecutive decided windows
            y0, y1 = good[i] * WIN, (good[i + 1] + 1) * WIN
            a, b = int(lv_w[good[i]]), int(lv_w[good[i + 1]])
            c = _corr(rows, g, ref, y0, y1)
            cur = 0
            for t in _levels(a, b):
                lvl_from = t - (1 if b > a else -1)
                k = _soft_step(c[cur:], lvl_from, t)
                ev.append((y0 + cur + k, t - lvl_from))
                cur += k
        rel[g] = ev
    if starts_known:
        return sorted((y, g, d) for g, ev in rel.items() for y, d in ev), refs
    return _assemble(rows, refs, rel, starts), refs


def _delta(rows, g, ref, y, tol=3, half=400):
    """Decided soft-level change of g vs ref across line y, or None."""
    (a, ma), (b, mb) = (
        _soft_level(_corr(rows, g, ref, lo, hi)) for lo, hi in ((y - half, y - tol), (y + tol, y + half))
    )
    return b - a if ma > MARGIN and mb > MARGIN else None


def _assemble(rows, refs, rel, starts, tol=3):  # noqa: C901
    """Relative per-group events -> absolute (line, group, delta).

    A drop in a reference shows as most of its dependants stepping -1 together: where >= 3 do so within `tol`, each
    dependant's step is measured across the line and, if most step -1, the drop is credited to the reference.
    Lags only accumulate, so each block is anchored with its least-lagged groups at 0; that minimum must be shared
    by >= 2 groups so one noisy start cannot set it.
    """
    events: list[tuple[int, int, int | None]] = []
    for ref in (REF_REFL, REF_THERM):
        deps = [g for g, r in refs.items() if r == ref and starts[g] is not None]
        events += [(0, g, None) for g, r in refs.items() if r == ref and starts[g] is None]
        if not deps:
            events.append((0, ref, None))
            continue
        neg = sorted((y, g) for g in deps for y, d in rel.get(g, []) if d == -1)
        cands: list[int] = []
        for y, _ in neg:
            near = [yy for yy, _ in neg if abs(yy - y) <= tol]
            if len(near) >= 3 and not any(abs(y - c) <= tol for c in cands):
                cands.append(int(np.median(near)))
        ref_drops = {}
        for y in cands:
            dl = {g: _delta(rows, g, ref, y) for g in deps}
            dec = [v for v in dl.values() if v is not None]
            if dec and sum(v == -1 for v in dec) > len(dec) / 2:
                ref_drops[y] = dl
        sv = sorted([starts[g] for g in deps] + [0])
        s0 = next(v for i, v in enumerate(sv) if i + 1 < len(sv) and sv[i + 1] == v) if len(sv) > 1 else 0
        if s0:
            events.append((0, ref, -s0))
        events += [(y, ref, 1) for y in ref_drops]
        for g in deps:
            if starts[g] - s0:
                events.append((0, g, starts[g] - s0))
            events += [(y, g, d) for y, d in rel.get(g, []) if not any(abs(y - r) <= tol for r in ref_drops)]
            for y, dl in ref_drops.items():
                d = dl[g] + 1 if dl[g] is not None else 0
                if d:
                    events.append((y, g, d))
    return _joint(events, rows, refs)


def _joint(events, rows, refs, tol=6, half=80, slack=0.05):
    """Snap drops within `tol` lines of each other to the one line that maximises their summed score.

    A group keeps its own line if the shared one costs it more than `slack` of its best score per line.
    """
    drops = sorted(e for e in events if e[0] > 0 and e[2] is not None and e[1] in refs)
    out = [e for e in events if e not in drops]
    clusters: list[list] = []
    for e in drops:
        if clusters and e[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    for cl in clusters:
        y = int(np.median([e[0] for e in cl]))
        y0 = max(0, y - half)
        scores = {}
        for e in cl:
            c = _corr(rows, e[1], refs[e[1]], y0, y + half)
            a, _ = _soft_level(c[: half - tol])
            b, _ = _soft_level(c[half + tol :])
            if a == b:
                b = a + e[2]
            scores[e] = _step_scores(c, a, b)
        kj = int(np.argmax(sum(scores.values())))
        for e, s in scores.items():
            k = kj if s.max() - s[kj] <= slack * 2 * half else int(np.argmax(s))
            out.append((y0 + k, e[1], e[2]))
    return sorted(out, key=lambda e: (e[0], e[1]))


def _step_scores(corr, a, b):
    ca, cb = corr[:, a + K], corr[:, b + K]
    return np.r_[0, np.cumsum(ca)] + np.r_[np.cumsum(cb[::-1])[::-1], 0]


# ---------------------------------------------------------------- table


def _runs(flag):
    """[(start, length)] of True runs."""
    d = np.diff(np.r_[0, flag.astype(int), 0])
    return [(int(a), int(b - a)) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True)]


def _fills(dup, zero, prof):
    """(kind, line, group, rows) for black and overwrite fills, only where the rest of the detector has signal."""
    others = np.median(prof[[g for g in range(NG) if g not in SKIP]], axis=0)
    lit = others > 0.05 * np.median(others)
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

    Columns kind,line,group,bands,value: `start`/`drop` give the lag change at that raw line (value None = start
    undetermined, taken as 0); `overwrite`/`black` give the number of raw rows from that line that hold no frame.
    The write is atomic, so tasks racing on one strip never read a partial table.
    """
    rows, prof, dup, zero = scan(src)
    events = detect(rows, prof)
    lines = ["kind,line,group,bands,value"]
    lines += [f"{'start' if y == 0 else 'drop'},{y},{g},{4 * g + 1}-{4 * g + 4},{d}" for y, g, d in events]
    lines += [f"{k},{y},{g},{4 * g + 1}-{4 * g + 4},{n}" for k, y, g, n in _fills(dup, zero, prof)]
    out = Path(fcsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, out)
    return out


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


# ---------------------------------------------------------------- radiance


def apply(rad, fcsv):
    """`rad` (band, y, x) with every group moved onto true frames, and the (band, y) mask of lost frames.

    `rad` is any y-range of the full strip the table was built on: its y coords are raw line numbers. A lost frame
    (dropped, copied or black, or sourced from outside `rad`) becomes NaN.
    """
    import dask.array as da
    import xarray as xr

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
    import dask.array as da

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
