import json
from types import SimpleNamespace

import pytest

from iirspy import solve

CFG0 = SimpleNamespace(min_match_frac=0.10)
CONSENSUS = (160.0, -8720.0)


def _fit(rejected, shift, accepted=True, match_frac=0.6):
    return {
        "quality": {"rejected": rejected, "match_frac": match_frac},
        "stats": {"coarse": {"total_shift_m": shift, "accepted": accepted}},
    }


def _outlier_strip(calls, seen):
    """Four chunks agreeing on `CONSENSUS`, except chunk 2 latches a spurious 15 km peak once."""

    def fake_solve_one(c, seed, *_):
        i = c["i"]
        calls.append((i, seed))
        seen[i] = seen.get(i, 0) + 1
        return {"chunk": c, "gcps": {}, "fit": _fit(False, (15000.0, 200.0) if i == 2 and seen[i] == 1 else CONSENSUS)}

    return fake_solve_one


def test_a_chunk_disagreeing_with_the_strip_is_reseeded_from_the_consensus(monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)
    monkeypatch.setattr(solve, "GROUP", "equatorial")
    calls, seen = [], {}
    monkeypatch.setattr(solve, "_solve_one", _outlier_strip(calls, seen))

    results, info = solve._solve_chunks([{"i": i} for i in range(4)], CFG0, None, None, None, 0.0, {}, False)

    assert info["coarse_consensus_m"] == list(CONSENSUS)
    assert info["consensus_enforced"] and not info["uncorrected"]
    # Only the outlier is re-solved, and only once; chunks already on the consensus are left alone.
    assert seen == {0: 1, 1: 1, 2: 2, 3: 1}
    assert calls[-1] == (2, CONSENSUS)
    assert [r["fit"]["reseeded"] for r in results] == [False, False, True, False]


def test_a_strip_with_no_corroborated_chunk_is_left_uncorrected(monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)
    monkeypatch.setattr(solve, "GROUP", "equatorial")
    calls = []

    def fake_solve_one(c, seed, *_):
        calls.append(c["i"])
        # Coarse accepted, but the match rate never corroborated it -- so it cannot vote.
        return {"chunk": c, "gcps": {}, "fit": _fit(True, (9920.0, -320.0), match_frac=0.004)}

    monkeypatch.setattr(solve, "_solve_one", fake_solve_one)
    _, info = solve._solve_chunks([{"i": i} for i in range(3)], CFG0, None, None, None, 0.0, {}, False)

    assert info["uncorrected"] and info["coarse_consensus_m"] is None
    assert not info["consensus_enforced"]
    assert calls == [0, 1, 2]


def test_polar_records_the_consensus_without_reseeding(monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)
    monkeypatch.setattr(solve, "GROUP", "south")
    monkeypatch.setattr(solve, "CONSENSUS_ALL_GROUPS", False)
    calls, seen = [], {}
    monkeypatch.setattr(solve, "_solve_one", _outlier_strip(calls, seen))

    results, info = solve._solve_chunks([{"i": i} for i in range(4)], CFG0, None, None, None, 0.0, {}, False)

    assert info["coarse_consensus_m"] == list(CONSENSUS)
    assert not info["consensus_enforced"]
    assert seen == {0: 1, 1: 1, 2: 1, 3: 1}  # nothing re-solved
    assert results[2]["fit"]["shift_dev_m"] > solve.CONSENSUS_TOL_M  # but the disagreement is on record


def test_build_l1_cache_hit_requires_matching_calibrate_kwargs(tmp_path, monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)

    def boom(*args, **kwargs):
        raise RuntimeError("rebuild-attempted")

    # Ancillary re-stage runs unconditionally even on a cache hit -- stub it out, and instead
    # detect a rebuild by the next real step past the cache check, `_stage_inputs`.
    monkeypatch.setattr(solve.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(solve, "_stage_inputs", boom)

    ftif = tmp_path / "l1.tif"
    ftif.write_bytes(b"stub")
    fmeta = ftif.with_suffix(".meta.json")
    fmeta.write_text(
        json.dumps({
            "scan0": 0,
            "lat_range": [10.0, 20.0],
            "bands": [1, 2],
            "calibrate_kwargs": {**solve.CALIBRATE_KWS, "empirical": True},
        })
    )

    # Matching kwargs (the default) -> cache hit, no rebuild attempted.
    assert solve.build_l1((10.0, 20.0), [1, 2], ftif) == (ftif, 0, (10.0, 20.0))

    # Different calibrate_kwargs to the same path -> cache miss, must not return the stale cube.
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        solve.build_l1((10.0, 20.0), [1, 2], ftif, calibrate_kwargs={"empirical": False})

    # Pre-existing meta.json with no recorded calibrate_kwargs (written before this check existed)
    # is treated as unknown, not as "assume production defaults" -> also rebuilds.
    fmeta.write_text(json.dumps({"scan0": 0, "lat_range": [10.0, 20.0], "bands": [1, 2]}))
    with pytest.raises(RuntimeError, match="rebuild-attempted"):
        solve.build_l1((10.0, 20.0), [1, 2], ftif)
