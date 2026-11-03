import glob
import os

import numpy as np
import pytest
import rasterio

from iirspy.iirs import L1

STAGE = os.environ.get("IIRS_ISSDC_CROP")


@pytest.mark.skipif(
    not STAGE, reason="set IIRS_ISSDC_CROP to an ISSDC tree holding an nci crop with its csv, label and spm"
)
@pytest.mark.filterwarnings("ignore:Dataset has no geotransform", "ignore:no thermal bands")
def test_nci_to_map_product_from_issdc_files_alone(tmp_path):
    """L1 -> georef(issdc) -> L2 -> to_geotiff, with no SPICE, DEM or arosics."""
    qub = glob.glob(f"{STAGE}/data/calibrated/*/ch2_iir_nci_*_d_img_*.qub")[0]
    sid = os.path.basename(qub).split("_")[3]
    l1 = L1(sid, STAGE)
    one = lambda p: glob.glob(f"{STAGE}/{p}")[0]
    geom = l1.georef(
        "issdc",
        group="equatorial",
        fgeom=one("geometry/calibrated/*/*.csv"),
        flabel=one("data/calibrated/*/*.xml"),
        fspm=one("miscellaneous/calibrated/*/*.spm"),
    )
    l2 = l1.calibrate("issdc", geom=geom)
    assert l2.img.shape == l1.img.shape and np.isfinite(l2.img.values).any()
    assert "photometric" in l2.img.attrs["provenance"]
    fout = l2.to_geotiff(tmp_path / "map.tif", geom=geom, grid=geom.grid(500.0))
    with rasterio.open(fout) as src:
        assert np.isfinite(src.read(1)).any()
