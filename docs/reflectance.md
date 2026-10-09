# L2 reflectance configuration

`L1.calibrate` converts L1 radiance to L2 reflectance (I/F, radiance divided by the solar flux and a photometric term). Each processing step is chosen by one field of `iirspy.l2config.L2Config`, and named presets bundle the fields for common methods. This page lists the steps, the presets, how to select and override them, and what is recorded on the output.

## Steps

The steps run in this order. Band numbers are IIRS band numbers, 1-indexed (band 1 is 712.3 nm, band 256 the last).

1. Solar flux: `flux` selects the solar spectrum and `sdist` scales it by the squared Sun distance.
2. Thermal removal: `thermal` selects the model, `thermal_eps` is the emissivity, and `thermal_bands` is the inclusive `(first, last)` band-number range whose mean brightness temperature sets the pixel temperature. The emitted radiance `thermal_eps * B(T)` is subtracted at every band.
3. Clip: `clip_negative` sets negative thermal-removed radiance to zero.
4. Polish: `polish` divides each band by a per-band factor, mapped by band number. Bands absent from the table get 1.0.
5. Photometric normalization: `photom` is the disk function, `topo` enables the per-scene topography product, `mu_min` floors the cosines of the incidence and emission angles, `min_lit` nulls pixels with a lit fraction below it (needs topo), and `phase` rescales onto Besse's 30 degree phase reference.
6. Spectral smoothing: `smoothing` is the moving-average width in bands (0 is off) and `smoothing_edges` sets the end bands. The value `"pair"` follows the CH2IIRS QGIS plugin and is defined for width 3 only. The first band is the mean of bands 2 and 3, and the last band is the mean of the last two bands. The other values are `"nan"` and `"reflect"`.

`L2._compute_reflectance` folds steps 1, 3, 4 and 5 into one division, `clip(rad - thermal) / (photometric * flux * polish)`. This is mathematically identical to applying them in the plugin's order, because clip and polish commute with the positive divisors.

## Presets

Table 1: The L2 presets and their field values. Fields not listed keep the `L2Config` defaults.

| Preset | Non-default fields | Purpose |
|---|---|---|
| `iirspy` (default) | `thermal="isothermal"` | Current best method of this package. Sun distance from SPICE, isothermal thermal removal at emissivity 1 over bands 227-247, Lambert disk function, topography used when supplied. |
| `issdc` | `sdist="none"`, `thermal="isothermal"`, `clip_negative=True`, `polish="issdc_ndi"`, `topo=False`, `smoothing=3` | Reproduces the ISSDC PRADAN NDI L2 product: no distance scaling, emissivity 1 thermal removal over bands 227-247, clip, `issdc_ndi` polish, no topography, 3-band smoothing. |
| `verma_plugin` | `sdist="none"`, `thermal="isothermal"`, `thermal_eps=0.95`, `clip_negative=True`, `polish="verma2022"`, `topo=False`, `smoothing=3` | The CH2IIRS QGIS plugin settings of Verma et al. (2022), for comparison with plugin output. |

The `issdc` preset was compared with NDI on 9 windows from 3 scenes. The median ratio to NDI over 750-2300 nm was within 0.2% in each window, and the per-band RMS difference was about 1.1% to 1.4%. The test `tests/test_l2config.py::test_issdc_preset_reproduces_ndi` asserts a 1% tolerance per window and is skipped when the local `l2_repro` window cache is absent. Three scenes do not establish agreement for other scenes, illumination or terrain.

## Usage

A preset name is the first argument of `calibrate`. The call needs an L1 object with its SPM file.

```python
>>> import iirspy
>>> l1 = iirspy.L1("20210720T2333026105", "/path/to/data/")  # doctest: +SKIP
>>> l2 = l1.calibrate()  # default preset "iirspy"  # doctest: +SKIP
>>> l2 = l1.calibrate("issdc")  # doctest: +SKIP
```

Keyword arguments other than `inc`, `dem`, `topo` and `sun_az_offset` override single fields of the preset.

```python
>>> l2 = l1.calibrate("issdc", sdist="spice")  # doctest: +SKIP
>>> l2 = l1.calibrate("issdc", flux="noaa_cdr_monthly")  # doctest: +SKIP
```

The name `topo` is both an `L2Config` field and a `calibrate` keyword, and the keyword takes it. To enable topography with another disk function, build the config explicitly and pass the topography file as the keyword.

```python
>>> from iirspy.l2config import L2Config
>>> cfg = L2Config.preset("issdc", topo=True, photom="lunar_lambert")
>>> l2 = l1.calibrate(cfg, topo="topo.tif")  # doctest: +SKIP
```

Per-scene inputs are keyword arguments of `calibrate`. They are `inc` (solar incidence in degrees, per line or per pixel, default from the SPM file), `dem`, `topo` (camera-space slope, aspect and optionally lit fraction, see `iirspy.photometry.load_topo`), and `sun_az_offset` (degrees added to the SPM solar azimuth to reach the topography grid-north frame). They are not config fields, so they are not recorded in `l2_config`.

A config can be built without data. Presets are plain dictionaries applied over the defaults.

```python
>>> L2Config.preset("issdc").to_attrs()["smoothing"]
3
>>> L2Config.preset("issdc", polish=None).polish is None
True
>>> L2Config().thermal_bands
(227, 247)
>>> from iirspy.l2config import PRESETS
>>> sorted(PRESETS)
['iirspy', 'issdc', 'verma_plugin']

```

An unknown preset, field or field value raises `ValueError` or `TypeError` when the config is built.

## Command line

The command `iirs-compute-refl <sid> --group <g> --l2 PRESET --l2-<field> VALUE ...` computes reflectance for a scene. There is one `--l2-<field>` flag per `L2Config` field, with underscores replaced by hyphens. Each value is parsed as JSON and, if that fails, used as a string.

```
iirs-compute-refl 20210720T2333026105 --group g1 --l2 issdc \
    --l2-thermal-bands [225,246] --l2-polish null --l2-topo false
```

This command ships with the georeferencing tools on the `georef` branch.

## Recorded metadata

Each product carries the configuration that made it, so a result can be reproduced from the file alone.

- `refl.attrs["l2_config"]` is a JSON string of every `L2Config` field. A callable `photom` is recorded by its function name only, so such a config cannot be rebuilt.
- Per-scene diagnostics in `refl.attrs`: `solar_distance_au`, `inc_min_deg`, `inc_max_deg`, `inc_mean_deg`, `mu0_floor_frac` and `mu_floor_frac` (fractions of pixels below `mu_min`), `topo_used`, and `sun_az_offset`.
- Coordinate `T` [K] is the per-pixel temperature, present when `thermal` is not `"none"`.
- The CLI writes `refl_summary.json` with the keys `l2_preset`, `l2_config` and `l2_diagnostics`.

```python
>>> import json
>>> cfg = L2Config(**json.loads(l2.attrs["l2_config"]))  # doctest: +SKIP
>>> l2.attrs["solar_distance_au"], l2.attrs["inc_mean_deg"]  # doctest: +SKIP
>>> l2["T"].median().item()  # median pixel temperature [K]  # doctest: +SKIP
```

The round trip also works for any config without data, because `to_attrs` returns JSON-safe values.

```python
>>> cfg = L2Config.preset("verma_plugin")
>>> L2Config(**json.loads(json.dumps(cfg.to_attrs()))) == cfg
True

```

## Shipped data files

Table 2: Data files read by the L2 steps, both under `iirspy/data/`.

| File | Used by | Content |
|---|---|---|
| `spectral_polish_issdc_ndi.csv` | `polish="issdc_ndi"` | Columns `band`, `polish`, `ci95_lo`, `ci95_hi`, `n_scenes`, `source`, one row per band. |
| `solar_flux_noaa_cdr_v03.csv` | `flux="noaa_cdr_mean"`, `"noaa_cdr_monthly"` | Column `band`, `mean_2010_2019`, and one `YYYY-MM` column per month. Units are mW/cm^2/um at 1 AU, convolved to the IIRS bands. |

The NOAA monthly lookup takes the month from the start time in the scene label. A month absent from the table raises `ValueError`. The table's last column is 2025-12.

### Untrusted bands in the issdc_ndi table

A band with `n_scenes` 0 and `polish` 1.0 is untrusted. It has no measured correction, so it receives no polish. In `issdc_ndi` the untrusted bands are 1 to 5 and 167 to 256 (3509.8 nm and longer). This includes the thermal bands 227 to 247. The 161 trusted bands, 6 to 166, have a median polish of 1.113 (range 1.044 to 1.183).

The trusted bands are divided by about 1.11 and the untrusted bands are not. The product therefore has a step of about 11% at the boundary between band 166 (3493.0 nm) and band 167, near 3.5 um. Reflectance should not be compared across that boundary, for example in band ratios or spectral slopes spanning it.
