import dask.array as da
import numpy as np
import xarray as xr

from iirspy.l1 import frame_defects


def test_detect_recovers_starts_and_drops():
    """Reflected groups start one frame behind the thermal ones (a negative start offset), g20 and REF drop later (lines within 1)."""
    rng = np.random.default_rng(1)
    nl, nx = 6000, 110
    field = rng.standard_normal((nl + 40, nx))
    y = np.arange(nl)

    def lag(g):
        return int(g < 43) + (y >= 1500) * int(g == 20) + (y >= 4500) * int(g == frame_defects.REF)

    rows = np.zeros((frame_defects.NG, nl, nx), np.float16)
    for g in range(frame_defects.NG):
        f = field[y + 20 + lag(g)] + 0.3 * rng.standard_normal((nl, nx))
        f -= f.mean(1, keepdims=True)
        rows[g] = f / np.linalg.norm(f, axis=1, keepdims=True)
    got = frame_defects.detect(rows)
    live = [g for g in range(frame_defects.NG) if g not in frame_defects.SKIP and g < 43]
    want = sorted([(0, g, 1) for g in live] + [(1500, 20, 1), (4500, frame_defects.REF, 1)])
    assert [e[1:] for e in got] == [e[1:] for e in want]
    assert max(abs(g[0] - w[0]) for g, w in zip(got, want, strict=True)) <= 1


def test_a_strip_reads_the_db_table_of_the_collect_it_reads(tmp_path):
    """A strip held by two data centers resolves to one zip, and the db hands back that zip's table."""
    import zipfile

    from iirspy import io
    from iirspy.cli._run import FRAME_DEFECT_DB, Job, ensure_frame_defects
    from iirspy.georef import chunks

    sid = "20260701T1835107378"
    (tmp_path / "zips").mkdir()
    db = tmp_path / FRAME_DEFECT_DB
    db.parent.mkdir(parents=True)
    with zipfile.ZipFile(db, "w") as z:
        for dc in ("d18", "d32"):
            (tmp_path / "zips" / f"ch2_iir_nri_{sid}_d_img_{dc}.zip").touch()
            z.writestr(f"{sid}_{dc}_frame_defects.csv", dc)
    paths = io.Paths(archive=tmp_path)
    job = Job(sid, "south", chunks.nri_zip(sid, paths), tmp_path, tmp_path / "log", paths, tmp_path)
    assert job.zip.name.endswith("_d32.zip")
    assert ensure_frame_defects(job, tmp_path / "keep").read_text() == "d32"


def test_spatial_fill_bridges_holes_up_to_span_on_either_axis():
    """A plane in x and y comes back exactly through dead columns, lost rows and a blob; a hole wider than SPAN on both
    axes stays NaN, also across a dask chunk edge."""
    y, x = np.mgrid[0:40, 0:30]
    truth = np.stack([1 + 0.1 * x + 0.05 * y, 2 + 0.03 * x - 0.02 * y])
    a = truth.copy()
    a[:, :18, 10:12] = np.nan  # two dead columns (rows 0-17)
    a[:, 20:23, :] = np.nan  # three lost whole frames, across the chunk edge at y=21
    a[:, 30:32, 3:5] = np.nan  # 2x2 blob
    a[:, 33:39, 20:26] = np.nan  # 6x6: beyond SPAN on both axes
    rad = xr.DataArray(da.from_array(a, chunks=(2, 21, 30)), dims=("band", "y", "x"), coords={"band": [1, 2]})
    out = frame_defects.fill_gaps(rad, interp_spatial=True).values
    wide = np.zeros(a.shape, bool)
    wide[:, 33:39, 20:26] = True
    np.testing.assert_allclose(out[~wide], truth[~wide])
    assert np.isnan(out[wide]).all()
