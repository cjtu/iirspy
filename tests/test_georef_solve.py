"""Scene solve: consensus reseeding and drift handling of chunk offsets, L1 cache reuse, product landing."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iirspy import io
from iirspy.cli import _run
from iirspy.georef import arosics
from iirspy.l1.frames import table_key

WS = Path(".")
CFG0 = SimpleNamespace(min_match_frac=0.10)
CONSENSUS = (160.0, -8720.0)
SPAN = 130_000.0  # m, a typical chunk's along-track span


def _chunks(n):
    return [{"i": i, "s0": i * SPAN, "s1": (i + 1) * SPAN, "t": (0.0, 1.0)} for i in range(n)]


def _fit(rejected, shift, accepted=True, match_frac=0.6):
    return {
        "quality": {"rejected": rejected, "match_frac": match_frac},
        "stats": {"coarse": {"total_shift_m": shift, "accepted": accepted}},
    }


def _outlier_strip(calls, seen):
    """Four chunks agreeing on `CONSENSUS`, except chunk 2 latches a spurious 15 km peak once."""

    def fake_solve_one(work, sid, group, frames, kernels, c, seed, *_, pin_coarse=False):
        i = c["i"]
        calls.append((i, seed, pin_coarse))
        seen[i] = seen.get(i, 0) + 1
        return {"chunk": c, "gcps": {}, "fit": _fit(False, (15000.0, 200.0) if i == 2 and seen[i] == 1 else CONSENSUS)}

    return fake_solve_one


def test_a_chunk_disagreeing_with_the_strip_is_reseeded_from_the_consensus(monkeypatch):
    calls, seen = [], {}
    monkeypatch.setattr(arosics, "_solve_one", _outlier_strip(calls, seen))

    results, info = arosics.solve_chunks(
        WS, "sid", "equatorial", None, [], _chunks(4), CFG0, None, None, None, 0.0, {}, False
    )

    assert all(v == list(CONSENSUS) for v in info["coarse_consensus_m"].values())
    assert info["consensus_outlier_anchors"] == {"along": [2], "cross": [2]} and not info["uncorrected"]
    # Only the outlier is re-solved, and only once; chunks already on the consensus are left alone.
    assert seen == {0: 1, 1: 1, 2: 2, 3: 1}
    # Pinned: the re-solve may not run its own coarse search and walk back off the consensus.
    assert calls[-1] == (2, CONSENSUS, True)
    assert [r["fit"]["reseeded"] for r in results] == [False, False, True, False]


def test_a_strip_with_no_corroborated_chunk_is_left_uncorrected(monkeypatch):
    calls = []

    def fake_solve_one(work, sid, group, frames, kernels, c, seed, *_):
        calls.append(c["i"])
        # Coarse accepted, but the match rate never corroborated it -- so it cannot vote.
        return {"chunk": c, "gcps": {}, "fit": _fit(True, (9920.0, -320.0), match_frac=0.004)}

    monkeypatch.setattr(arosics, "_solve_one", fake_solve_one)
    _, info = arosics.solve_chunks(
        WS, "sid", "equatorial", None, [], _chunks(3), CFG0, None, None, None, 0.0, {}, False
    )

    assert info["uncorrected"] and info["coarse_consensus_m"] is None
    assert calls == [0, 1, 2]


def test_polar_strips_are_reseeded_too(monkeypatch):
    """Polar shifts are near-constant in their own grid frame (run 2: p50 0.40 km per strip)."""
    calls, seen = [], {}
    monkeypatch.setattr(arosics, "_solve_one", _outlier_strip(calls, seen))

    results, _ = arosics.solve_chunks(WS, "sid", "south", None, [], _chunks(4), CFG0, None, None, None, 0.0, {}, False)

    assert seen == {0: 1, 1: 1, 2: 2, 3: 1}
    assert results[2]["fit"]["shift_dev_m"] == 0.0


def _replay(monkeypatch, strip, s0, t):
    """Run `solve_chunks` over recorded (match_frac, total_shift_m) per chunk, planned at `s0` [km] along unit
    track `t`; a pinned re-solve lands on its seed."""
    pinned = {}

    def fake_solve_one(work, sid, group, frames, kernels, c, seed, *_, pin_coarse=False):
        frac, shift = strip[c["i"]]
        if pin_coarse:
            pinned[c["i"]] = seed
            shift = seed
        return {"chunk": c, "gcps": {}, "fit": _fit(frac < 0.10, shift, True, frac)}

    monkeypatch.setattr(arosics, "_solve_one", fake_solve_one)
    chunks = [{"i": i, "s0": v * 1e3, "s1": v * 1e3 + 138_000.0, "t": t} for i, v in enumerate(s0)]
    return pinned, arosics.solve_chunks(WS, "sid", "south", None, [], chunks, CFG0, None, None, None, 0.0, {}, False)[1]


def test_a_line_timing_drift_is_followed_along_track(monkeypatch):
    """a3_9regions run 2 `20201226T1745264921` south, verbatim: ~45 m/km of along-track drift, cross-track <=0.7 km.

    Its equatorial group never registers (the drift walks it out of coarse capture), and chunk 3 matched
    0.168 at its own shift vs 0.018 when pinned to a grid-frame consensus capped at 12 m/km.
    """
    strip = [
        (0.196, (-2200.0, -4200.0)),
        (0.318, (0.0, -800.0)),
        (0.179, (300.0, 800.0)),
        (0.168, (4800.0, 7400.0)),
        (0.079, (8200.0, 10900.0)),
        (0.112, (12500.0, 17400.0)),
        (0.120, (16800.0, 23000.0)),
    ]
    pinned, info = _replay(monkeypatch, strip, [0, 126, 252, 381, 504, 628, 752], (0.5613, 0.8276))
    assert info["consensus_outlier_anchors"]["cross"] == []
    # Chunk 2 is 4.3 km off the along-track line but matched as well as the anchors under it: kept.
    assert info["consensus_outlier_anchors"]["along"] == [2]
    assert sorted(pinned) == [4]  # the one rejected chunk, onto the along-track line between its neighbours
    along = 0.5613 * pinned[4][0] + 0.8276 * pinned[4][1]
    assert 8818 < along < 21416


def test_a_genuine_drift_is_followed_not_reseeded(monkeypatch):
    """a3_9regions run 2 `20210103T0245339065` south, verbatim: 5.6 km of smooth drift (7.4 m/km)."""
    strip = [
        (0.512, (8500.0, -4300.0)),
        (0.769, (7400.0, -3700.0)),
        (0.848, (6600.0, -3200.0)),
        (0.632, (5800.0, -2900.0)),
        (0.684, (4800.0, -2200.0)),
        (0.626, (3500.0, -1600.0)),
        (0.395, (2900.0, -1100.0)),
    ]
    pinned, info = _replay(monkeypatch, strip, [0, 128, 257, 388, 513, 637, 761], (-0.8893, 0.4573))
    assert pinned == {} and info["consensus_outlier_anchors"] == {"along": [], "cross": []}


def test_a_registered_chunk_anchors_even_when_its_coarse_peak_was_refused(monkeypatch):
    """E1 `20191217T2335209162`, verbatim: (match_frac, accepted, total_shift_m) per chunk.

    Only the two mis-latched chunks (match_frac 0.004, 10-15 km out) had `coarse.accepted`; the one
    chunk that actually registered (ch6, 0.167) refused its peak and kept its zero seed. Requiring
    both left the strip with no anchor and shipped the 15 km chunks uncorrected.
    """
    strip = [
        (0.013, False, (0.0, 0.0)),
        (0.012, False, (0.0, 0.0)),
        (0.008, False, (0.0, 0.0)),
        (0.004, True, (14880.0, 160.0)),
        (0.044, False, (0.0, 0.0)),
        (0.004, True, (-10240.0, -640.0)),
        (0.167, False, (0.0, 0.0)),
    ]
    pinned = []

    def fake_solve_one(work, sid, group, frames, kernels, c, seed, *_, pin_coarse=False):
        frac, accepted, shift = strip[c["i"]]
        if pin_coarse:
            pinned.append(c["i"])
            shift, accepted = seed, True
        return {"chunk": c, "gcps": {}, "fit": _fit(frac < 0.10, shift, accepted, frac)}

    monkeypatch.setattr(arosics, "_solve_one", fake_solve_one)
    results, info = arosics.solve_chunks(
        WS, "sid", "equatorial", None, [], _chunks(7), CFG0, None, None, None, 0.0, {}, False
    )

    assert all(v == [0.0, 0.0] for v in info["coarse_consensus_m"].values()) and not info["uncorrected"]
    assert pinned == [3, 5]  # the two mis-latched chunks, and only those
    assert all(r["fit"]["shift_dev_m"] == 0.0 for r in results)


def test_build_l1_cache_hit_requires_matching_calibrate_kwargs(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("rebuild-attempted")

    # Ancillary re-stage runs unconditionally even on a cache hit -- stub it out, and instead
    # detect a rebuild by the next real step past the cache check, `stage_inputs`.
    monkeypatch.setattr(_run.io, "extract", lambda *a, **k: [])
    monkeypatch.setattr(_run, "stage_inputs", boom)
    monkeypatch.setattr(_run.ck, "ancillary", lambda sid, paths: {"geometry/calibrated": None})

    ffr = tmp_path / "frames.csv"
    ffr.write_text("kind,line,group,bands,value\n")
    paths = io.Paths(tmp_path, (tmp_path,), (), tmp_path, tmp_path)
    job = _run.Job(
        "20201202T2319552644", "south", tmp_path / "x.zip", tmp_path / "stage", tmp_path / "x.log", paths, tmp_path, ffr
    )
    ftif = tmp_path / "l1.tif"
    ftif.write_bytes(b"stub")
    fmeta = ftif.with_suffix(".meta.json")
    fmeta.write_text(
        json.dumps({
            "scan0": 0,
            "lat_range": [10.0, 20.0],
            "bands": [1, 2],
            "calibrate_kwargs": {**_run.CALIBRATE_KWS, "frame_offsets": str(ffr), "empirical": True},
            "frames": table_key(ffr),
            "crop_geom": "label",
        })
    )

    # Matching kwargs (the default) -> cache hit, no rebuild attempted.
    assert _run.build_l1(job, (10.0, 20.0), [1, 2], ftif) == (ftif, 0, (10.0, 20.0))

    # Cropped with a different geometry csv than the solve now uses -> cache miss.
    monkeypatch.setattr(_run.ck, "ancillary", lambda sid, paths: {"geometry/calibrated": tmp_path / "x_g_grd_d18.csv"})
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        _run.build_l1(job, (10.0, 20.0), [1, 2], ftif)
    monkeypatch.setattr(_run.ck, "ancillary", lambda sid, paths: {"geometry/calibrated": None})

    # Different calibrate_kwargs to the same path -> cache miss, must not return the stale cube.
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        _run.build_l1(job, (10.0, 20.0), [1, 2], ftif, calibrate_kwargs={"empirical": False})

    # A different frame table -> cache miss.
    ffr.write_text("kind,line,group,bands,value\ndrop,5,3,13-16,1\n")
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        _run.build_l1(job, (10.0, 20.0), [1, 2], ftif)

    # Pre-existing meta.json with no recorded calibrate_kwargs (written before this check existed)
    # is treated as unknown, not as "assume production defaults" -> also rebuilds.
    fmeta.write_text(json.dumps({"scan0": 0, "lat_range": [10.0, 20.0], "bands": [1, 2]}))
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        _run.build_l1(job, (10.0, 20.0), [1, 2], ftif)


def test_keep_products_lands_only_what_is_new(tmp_path):
    out, dest = tmp_path / "out", tmp_path / "keep"
    out.mkdir()
    (out / "s_equatorial.gcps").write_text("gcps")
    assert [f.name for f in arosics.keep_products("s", "equatorial", out, dest)] == ["s_equatorial.gcps"]
    (out / "s_equatorial_glt.tif").write_text("glt")
    assert [f.name for f in arosics.keep_products("s", "equatorial", out, dest)] == ["s_equatorial_glt.tif"]
    assert arosics.keep_products("s", "equatorial", out, dest) == []
    (out / "s_equatorial.gcps").write_text("re-solved")
    assert [f.name for f in arosics.keep_products("s", "equatorial", out, dest)] == ["s_equatorial.gcps"]
    assert (dest / "s_equatorial.gcps").read_text() == "re-solved"
