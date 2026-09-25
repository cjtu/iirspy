# Using an iirspy product

IIRS products are written in **camera space** and projected on demand through a **GLT** 
(geographic lookup table) which was generated through feature matching to LOLA dem. 

## What a scene gives you

| file | what it is |
|---|---|
| `<sid>_<group>_l1_rad.tif` / `_l2_refl.tif` | the product, camera space, float32, NaN nodata |
| `geometry/recalibrated/<day>/<sid>_<group>/<sid>_<group>.gcps` | the solved control points the two below from |
| `geometry/recalibrated/<day>/<sid>_<group>/<sid>_<group>_glt.tif` | 2-band int32 lookup table: where each map pixel comes from |
| `<sid>_<group>_l2_refl.vrt` | sidecar carrying the scene's GCPs + CRS, for tools that warp |

Here, `group` is `south`, `north` or `equatorial`. A strip crossing more than one gets a GLT per group,
and **all of them address the same product file** — see [absolute scans](#conventions).

## Which one do I use?

**Use the GLT.** It is exact, needs no parameters, and applying it is array indexing. The `.vrt` is
only for tools that cannot consume a lookup table.

## Is the TPS warp baked in?

**In the GLT, yes.** The table *is* the result of a thin-plate-spline warp at zero error tolerance
with nearest-neighbour sampling. Those choices are already frozen into its integer indices, so
applying it takes no warp settings at all and cannot be got wrong.

**In the VRT, no.** The VRT carries only GCPs. Any tool you hand it to must be told to use TPS at
zero tolerance with nearest neighbour, or it will silently fit a *polynomial* at GDAL's default
0.125 px tolerance. Measured on a 300 x 300 window of `20201202T2319552644` south:

| VRT warped with | vs the GLT |
|---|---|
| `-tps -et 0 -r near` | identical, 0 of 88376 pixels differ |
| GDAL defaults | **80% of pixels differ** (70850/88376), median difference 5.47 |

So always pass all three:

```
-tps -et 0 -r near
```

## Python

```python
import xarray as xr
from iirspy import georef

table, tags = georef.read_glt(
    "geometry/recalibrated/20201202/20201202T2319552644_south/20201202T2319552644_south_glt.tif"
)
cube = xr.open_dataarray("20201202T2319552644_south_l2_refl.tif", engine="rasterio")

projected = georef.apply_glt(cube.values, table, cube_scan0=int(cube.y.values[0]))
```

`projected` is `(band, y, x)` on the GLT's grid, NaN outside the footprint. A whole scene is ~100 GB
at 222 bands, so in practice take a window:

```python
sub = georef.apply_glt(cube.values, table, cube_scan0=int(cube.y.values[0]), window=(r0, r1, c0, c1))
```

To write a projected GeoTIFF for an area of interest:

```python
import rasterio
from iirspy import georef

(ny, nx), transform = georef.window_of(cfg, window)          # cfg from the solve, or the GLT tags
with rasterio.open(
    "aoi.tif", "w", driver="GTiff", height=ny, width=nx, count=sub.shape[0], dtype="float32",
    nodata=float("nan"), crs=georef.stereo_crs(tags["group"]), transform=transform, compress="LZW",
) as dst:
    dst.write(sub)
```

Band numbers and wavelengths ride in the product's tags (`band_numbers`, and a `Band_N` tag per
plane); band descriptions are the wavelength in nm.

## GDAL

Either apply the GLT yourself — it is two integer rasters, so the indexing above works in any
language — or warp the VRT:

```bash
gdalwarp -tps -et 0 -r near -t_srs "$(gdalsrsinfo -o wkt geometry/recalibrated/.../..._glt.tif)" \
         20201202T2319552644_south_l2_refl.vrt projected.tif
```

Omitting `-tps` or `-et 0` changes the geometry. Warping the whole scene writes the full ~100 GB;
add `-te <xmin> <ymin> <xmax> <ymax>` to cut it to an area of interest first.

## QGIS

QGIS reads the camera-space `.tif` as an unreferenced raster — useful for inspecting the detector,
not for mapping. To get it on the map:

- **Processing → GDAL → Raster projections → Warp (reproject)**, source the `.vrt`, resampling
  *Nearest neighbour*, and put `-tps -et 0` in **Additional command-line parameters**. The
  transformation is not exposed in the dialog, so without that box QGIS gives you the polynomial.
- For anything interactive, pre-render a projected GeoTIFF for your AOI in Python and load that —
  a TPS warp with several thousand GCPs is far too slow to re-run on every pan.

## ENVI

ENVI supports GLTs natively (*Georeference from GLT*), but its convention differs from ours, so
convert first:

| | iirspy | ENVI |
|---|---|---|
| band 1 | camera column, 0-based | sample, 1-based |
| band 2 | **absolute scan**, 0-based | line in the file, 1-based |
| outside footprint | `-1` | `0` |

```python
envi = table.copy()
inside = envi[0] >= 0
envi[1] -= cube_scan0   # absolute scan -> row in this file
envi[0] += 1            # 0-based -> 1-based
envi[1] += 1
envi[:, ~inside] = 0    # nodata -1 -> 0
```

This conversion is untested against ENVI itself; check one scene before trusting a batch. The
simplest ENVI route is to export a projected GeoTIFF from Python and open that.

## Conventions

**Camera space and absolute scans.** Row 0 of a product is not necessarily scan 0 of the strip: a
`south` group starts at the pole, but an `equatorial` or midlat group starts partway down. Products
record their first scan — GeoTIFF in the geotransform, ENVI as `y start` — and the GLT's band 2
holds **absolute** scans, which is why `apply_glt` asks for `cube_scan0`. Get it from the product:

```python
cube_scan0 = int(xr.open_dataarray(product, engine="rasterio").y.values[0])
```

Because both are absolute, one downloaded cube serves every group's table, and you never have to
reproduce the crop the solve used.

**CRS and grid.** `stereo_crs(group)` at 40 m/px. The GLT carries the CRS and transform, so it also
defines the output grid; the products themselves carry no map CRS.

**Nodata.** `-1` in the GLT means *no camera pixel lands on this map pixel*, never *that pixel had
no data*. Per-band NaN comes from the product at apply time, so one table serves L1, L2 and any
derived product.

**Units.** L1 radiance is W/m²/sr/µm; L2 is unitless I/F. Both carry `name`, `units` and the
calibration diagnostics (`photom`, `solar_distance_au`, `inc_*_deg`, `empirical_notes`) as tags.
