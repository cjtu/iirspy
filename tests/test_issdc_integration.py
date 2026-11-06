"""Test pipeline reproduces ISSDC data: L1 -> L2 -> map product, and issdc preset vs official NDI."""

import glob
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from iirspy import instrument
from iirspy.georef import Geometry
from iirspy.georef.geometry import quality
from iirspy.iirs import L1
from iirspy.l2 import calibrate as calibrate_l2
from iirspy.l2.config import L2Config

STAGE = os.environ.get("IIRS_ISSDC_CROP")
L2_REPRO = os.environ.get("IIRS_L2_REPRO")  # cached NCI/NDI windows, see test_issdc_preset_reproduces_ndi


@pytest.mark.skipif(
    not STAGE, reason="set IIRS_ISSDC_CROP to an ISSDC tree holding an nci crop with its csv, label and spm"
)
@pytest.mark.filterwarnings("ignore:Dataset has no geotransform", "ignore:no thermal bands")
def test_nci_to_map_product_from_issdc_files_alone(tmp_path):
    """L1 -> georef(issdc) -> L2 -> to_geotiff, with no SPICE, DEM or arosics."""
    qub = glob.glob(f"{STAGE}/data/calibrated/*/ch2_iir_nci_*_d_img_*.qub")[0]
    sid = os.path.basename(qub).split("_")[3]
    with rasterio.open(qub) as src:
        mid = src.height // 2
    l1 = L1(sid, STAGE, xyextent=(0, 249, mid, mid + 199))
    one = lambda p: glob.glob(f"{STAGE}/{p}")[0]
    geom = l1.georef(
        "issdc",
        group="equatorial",
        fgeom=one("geometry/calibrated/*/*.csv"),
        flabel=one("data/calibrated/*/*.xml"),
        fspm=one("miscellaneous/calibrated/*/*.spm"),
    )
    geom.save(tmp_path)
    geom = Geometry.open(tmp_path, sid, "equatorial")
    l2 = l1.calibrate("issdc", geom=geom, smoothing=1)
    assert l2.img.shape == l1.img.shape and np.isfinite(l2.img.values).any()
    assert "photometric" in l2.img.attrs["provenance"]
    fout = l2.to_geotiff(tmp_path / "map.tif", geom=geom, grid=geom.grid(500.0))
    with rasterio.open(fout) as src:
        assert np.isfinite(src.read()).any()


def _align(ndi, cands, pad, n, band=48):
    """The candidate window and line offset that best correlates with ndi's core lines."""
    ref = ndi[band, pad : pad + n, ::5].ravel()
    best = (-2.0, None)
    for rad in cands:
        for d in range(-pad, pad + 1):
            x = rad[band, pad + d : pad + d + n, ::5].ravel()
            ok = np.isfinite(ref) & np.isfinite(x)
            c = np.corrcoef(ref[ok], x[ok])[0, 1] if ok.sum() > 100 else -1.0
            if c > best[0]:
                best = (c, rad[:, pad + d : pad + d + n])
    return best[1]


@pytest.mark.skipif(not L2_REPRO, reason="set IIRS_L2_REPRO to the cached NCI/NDI window dir")
def test_issdc_preset_reproduces_ndi():
    """The issdc preset on official NCI radiance matches ISSDC NDI reflectance within 1% per window.

    IIRS_L2_REPRO holds <sid>/nci.npz (rad[, rad_flip]) and <sid>/ndi.npz (rfl, obs) windows of
    256 lines plus 8 padding lines each side.
    """
    pad, n = 8, 256
    wl = instrument.get_wls()
    short = (wl > 750) & (wl < 2300)
    meds = []
    for d in sorted(p for p in Path(L2_REPRO).iterdir() if (p / "nci.npz").exists() and (p / "ndi.npz").exists()):
        nci, ndi = np.load(d / "nci.npz"), np.load(d / "ndi.npz")
        rfl, sunzen = ndi["rfl"], ndi["obs"][:, 1]  # each npz key access re-reads the whole array
        rads = [nci[k] for k in ("rad", "rad_flip") if k in nci]
        for w in range(len(rfl)):
            ref = rfl[w][:, pad : pad + n]
            rad = _align(rfl[w], [r[w] for r in rads], pad, n)
            ny, nx = ref.shape[1:]
            coords = {"band": np.arange(1, 257), "wl": ("band", wl), "y": np.arange(ny), "x": np.arange(nx)}
            da = xr.DataArray(rad * instrument.RAD_NATIVE_SCALE, dims=("band", "y", "x"), coords=coords)
            da = da.assign_coords(solar_inc=("y", np.zeros(ny)), solar_az=("y", np.zeros(ny)))
            zen = sunzen[w, pad : pad + n]
            loc = {"lon": np.zeros(zen.shape), "lat": np.zeros(zen.shape), "radius": np.full(zen.shape, 1737400.0)}
            geom = Geometry.from_arrays(
                "sid",
                "south",
                0,
                "issdc",
                loc,
                {"sun_zenith": zen},
                np.zeros(zen.shape, "uint16"),
                quality("issdc", "warn"),
            )
            refl = calibrate_l2(da, L2Config.preset("issdc"), geom=geom).values
            ok = np.isfinite(refl) & np.isfinite(ref) & (ref > 0.01)
            ratio = np.where(ok, refl / np.where(ok, ref, 1), np.nan)
            meds.append(np.nanmedian(np.nanmedian(ratio.reshape(256, -1), 1)[short]))
    assert meds, f"no scene in {L2_REPRO} has both nci.npz and ndi.npz"
    np.testing.assert_allclose(meds, 1, atol=0.01)  # each window, median over 750-2300 nm
