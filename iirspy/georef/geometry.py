import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from iirspy.georef import backplanes as bp
from iirspy.georef.georef import GeorefConfig, glt_from_loc, to_stereo

GRADES = ("pass", "warn", "fail")


def quality(method: str, grade: str, residual_p95_m: float | None = None, frac_aligned: float | None = None) -> dict:
    if grade not in GRADES:
        raise ValueError(f"grade {grade!r} not in {GRADES}")
    return {"method": method, "grade": grade, "residual_p95_m": residual_p95_m, "frac_aligned": frac_aligned}


def _dataset(arrays: dict[str, np.ndarray], row0: int) -> xr.Dataset:
    ny, nx = next(iter(arrays.values())).shape
    return xr.Dataset(
        {k: (("y", "x"), np.asarray(v)) for k, v in arrays.items()},
        coords={"y": row0 + np.arange(ny) + 0.5, "x": np.arange(nx) + 0.5},
    )


@dataclass
class Geometry:
    sid: str
    group: str
    row0: int
    method: str
    quality: dict
    sidecars: dict[str, Path] = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)
    _loc: xr.Dataset | None = None
    _obs: xr.Dataset | None = None
    _qa: xr.DataArray | None = None

    @classmethod
    def from_arrays(
        cls,
        sid: str,
        group: str,
        row0: int,
        method: str,
        loc: dict[str, np.ndarray],
        obs: dict[str, np.ndarray],
        qa: np.ndarray,
        quality: dict,
        **kw,
    ) -> "Geometry":
        ny, nx = loc["lon"].shape
        nan = np.full((ny, nx), np.nan, "float32")
        obs = {n: np.asarray(obs.get(n, nan), "float32") for n in bp.OBS_NAMES}
        qa_da = _dataset({"qa_bits": np.asarray(qa, "uint16")}, row0)["qa_bits"]
        return cls(
            sid, group, row0, method, quality, _loc=_dataset(loc, row0), _obs=_dataset(obs, row0), _qa=qa_da, **kw
        )

    @property
    def stem(self) -> str:
        return f"{self.sid}_{self.group}"

    @property
    def loc(self) -> xr.Dataset:
        if self._loc is None:
            self._loc = bp.read_loc(self.sidecars["loc"])
        return self._loc

    @property
    def obs(self) -> xr.Dataset:
        if self._obs is None:
            self._obs = bp.read_obs(self.sidecars["obs"])
        return self._obs

    @property
    def qa(self) -> xr.DataArray:
        if self._qa is None:
            self._qa = bp.read_obs(self.sidecars["qa"])["qa_bits"]
        return self._qa

    @property
    def has_topo(self) -> bool:
        return "topo" in self.sidecars

    @property
    def shape(self) -> tuple[int, int]:
        ny, nx = self.loc.lon.shape
        return ny, nx

    def grid(self, res, bounds=None):
        """Export grid at `res` m/px in this group's CRS; `bounds` (xmin, ymin, xmax, ymax) default to the LOC's finite extent."""
        if bounds is None:
            x, y = to_stereo(self.group).transform(self.loc.lon.values, self.loc.lat.values)
            x, y = np.asarray(x), np.asarray(y)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                raise ValueError(f"{self.stem}: LOC has no finite (x, y)")
            bounds = (float(x[finite].min()), float(y[finite].min()), float(x[finite].max()), float(y[finite].max()))
        return GeorefConfig(pole=self.group, aoi=tuple(bounds), ps=res)

    def glt(self, grid, window=None):
        return glt_from_loc(self.loc, self.group, grid, window)

    def save(self, dirname) -> dict[str, Path]:
        d = Path(dirname)
        d.mkdir(parents=True, exist_ok=True)
        ny = self.shape[0]
        groups = {self.group: (self.row0, self.row0 + ny - 1)}
        tags = {
            "GEOMETRY_METHOD": self.method,
            "GEOMETRY_QUALITY": json.dumps(self.quality),
            **self.tags,
        }
        out = {
            "loc": bp.write_loc(
                d / f"{self.stem}_loc.tif",
                {**{k: self.loc[k].values for k in ("lon", "lat", "radius")}, "groups": groups},
                self.sid,
                row0=self.row0,
                tags=tags,
            ),
            "obs": bp.write_obs(
                d / f"{self.stem}_obs.tif",
                {
                    "bands": {n: self.obs[n].values for n in bp.OBS_NAMES},
                    "groups": groups,
                    "sensor_source": self.tags.get("SENSOR_SOURCE", "none"),
                },
                self.sid,
                row0=self.row0,
                tags=tags,
            ),
            "qa": bp.write_qa(
                d / f"{self.stem}_qa.tif",
                self.qa.values,
                self.sid,
                groups,
                {self.group: self.tags.get("SATURATION_CHECK", "not evaluated")},
                row0=self.row0,
                tags=tags,
            ),
        }
        summary = {
            "sid": self.sid,
            "group": self.group,
            "row0": self.row0,
            "method": self.method,
            "quality": self.quality,
            "sidecars": {k: str(v) for k, v in {**self.sidecars, **out}.items()},
            "provenance": self.provenance,
            "tags": self.tags,
        }
        fsum = d / f"{self.stem}_geometry.json"
        fsum.write_text(json.dumps(summary, indent=1, default=str))
        self.sidecars = {**self.sidecars, **out}
        return {**out, "summary": fsum}

    @classmethod
    def open(cls, dirname, sid: str, group: str) -> "Geometry":
        s = json.loads((Path(dirname) / f"{sid}_{group}_geometry.json").read_text())
        if (s["sid"], s["group"]) != (sid, group):
            raise ValueError(f"{dirname}: summary is for {s['sid']}/{s['group']}")
        return cls(
            s["sid"],
            s["group"],
            s["row0"],
            s["method"],
            s["quality"],
            {k: Path(v) for k, v in s["sidecars"].items()},
            s["provenance"],
            s["tags"],
        )
