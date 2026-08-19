"""
AROSICS tie-point matching, run in its own conda environment.

arosics needs `osgeo.gdal`, which PyPI does not ship, so the matcher cannot run in the iirspy
environment. Create its environment once, from the repository root:

    mamba env create -f arosics-environment.yml

`iirspy.georef.register` then calls this module as a subprocess in that environment whenever
arosics is not importable in the current one, and raises with the command above if the
environment does not exist. To run a solve by hand:

    conda run -n arosics python -m iirspy.coreg job.json

`job.json` holds the scene paths and the GeorefConfig fields; the solved GCP lattice and the
solve's stats are written as json to the path its "out" key names.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing iirspy

from iirspy.georef import GeorefConfig, register


def run_job(fjob):
    """Solve the registration `fjob` describes and write its GCPs and stats to job["out"]."""
    job = json.loads(Path(fjob).read_text())
    cfg = GeorefConfig(**job.get("cfg", {}))
    reg = register(
        job["ftif"],
        job["fgeom"],
        job["fspm"],
        cfg,
        kernels=job.get("kernels"),
        reference=job.get("reference"),
        verbose=job.get("verbose", False),
    )
    out = {"gcps": [[g.row, g.col, g.x, g.y] for g in reg.gcps], "stats": reg.stats}
    Path(job["out"]).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    run_job(sys.argv[1])
