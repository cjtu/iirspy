from iirspy import solve


def _fit(rejected, shift):
    return {"quality": {"rejected": rejected}, "stats": {"coarse": {"total_shift_m": shift}}}


def test_seeded_retry_backfills_from_the_nearest_accepted_neighbour(monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)
    calls = []

    def fake_solve_one(c, seed, *_):
        calls.append((c["i"], seed))
        i = c["i"]
        ok = seed == (20.0, 0.0) if i in (0, 1) else True
        shift = (30.0, 0.0) if i == 3 else (20.0, 0.0)
        return {"chunk": c, "gcps": {}, "fit": _fit(not ok, shift)}

    monkeypatch.setattr(solve, "_solve_one", fake_solve_one)
    chunks = [{"i": i} for i in range(4)]
    results = solve._solve_chunks(chunks, None, None, None, None, 0.0, {}, False)

    assert [r["chunk"]["i"] for r in results] == [0, 1, 2, 3]
    assert all(solve._accepted(r) for r in results)
    assert calls == [
        (0, (0.0, 0.0)),
        (1, (0.0, 0.0)),
        (2, (0.0, 0.0)),
        (3, (20.0, 0.0)),
        (1, (20.0, 0.0)),
        (0, (20.0, 0.0)),
    ]


def test_reverse_retry_stops_once_a_retry_still_fails(monkeypatch):
    monkeypatch.setattr(solve, "log", lambda msg: None)
    calls = []

    def fake_solve_one(c, seed, *_):
        calls.append(c["i"])
        ok = c["i"] == 2
        return {"chunk": c, "gcps": {}, "fit": _fit(not ok, (20.0, 0.0) if ok else (0.0, 0.0))}

    monkeypatch.setattr(solve, "_solve_one", fake_solve_one)
    chunks = [{"i": i} for i in range(3)]
    results = solve._solve_chunks(chunks, None, None, None, None, 0.0, {}, False)

    assert not solve._accepted(results[0])
    assert not solve._accepted(results[1])
    assert solve._accepted(results[2])
    assert calls == [0, 1, 2, 1]
