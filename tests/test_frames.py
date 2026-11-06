import numpy as np

from iirspy.l1 import frames


def test_detect_recovers_starts_and_drops():
    """Reflected groups start one frame behind the thermal ones (a negative start offset), g20 and REF drop later (lines within 1)."""
    rng = np.random.default_rng(1)
    nl, nx = 6000, 110
    field = rng.standard_normal((nl + 40, nx))
    y = np.arange(nl)

    def lag(g):
        return int(g < 43) + (y >= 1500) * int(g == 20) + (y >= 4500) * int(g == frames.REF)

    rows = np.zeros((frames.NG, nl, nx), np.float16)
    for g in range(frames.NG):
        f = field[y + 20 + lag(g)] + 0.3 * rng.standard_normal((nl, nx))
        f -= f.mean(1, keepdims=True)
        rows[g] = f / np.linalg.norm(f, axis=1, keepdims=True)
    got = frames.detect(rows)
    live = [g for g in range(frames.NG) if g not in frames.SKIP and g < 43]
    want = sorted([(0, g, 1) for g in live] + [(1500, 20, 1), (4500, frames.REF, 1)])
    assert [e[1:] for e in got] == [e[1:] for e in want]
    assert max(abs(g[0] - w[0]) for g, w in zip(got, want, strict=True)) <= 1
