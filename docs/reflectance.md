# Calibration configs

`L0.calibrate` converts raw counts to L1 radiance and `L1.calibrate` converts L1 radiance to L2 reflectance (I/F). Each method choice is one field of a frozen config, `iirspy.l1.config.L1Config` or `iirspy.l2.config.L2Config`. Named presets bundle the fields, and keyword overrides replace single fields.

```python
import iirspy

l0 = iirspy.L0("20210720T2333026105", "/path/to/data/")
l1 = l0.calibrate()                              # L1Config preset "iirspy"
geom = l1.georef("issdc", group="equatorial", fgeom=..., flabel=..., fspm=...)  # a Geometry, see below
l2 = l1.calibrate(geom=geom)                     # L2Config preset "iirspy"
l2 = l1.calibrate("issdc", geom=geom)            # reproduce the ISSDC L2 product
l2 = l1.calibrate("issdc", geom=geom, sdist="spice")  # ...with one field overridden
l2.to_geotiff("map.tif", geom=geom, grid=geom.grid(500.0))  # the only place the cube is map-projected
```

`iirspy.l1.calibrate` and `iirspy.l2.calibrate` run the steps in order. They are the only code that reads the config: each step function takes plain arguments, and `calibrate` skips a step whose fields are off. Each step that runs appends `{"step", "method"}` to the JSON list `attrs["provenance"]`, which a later level continues; the full config is in `l1_config`/`l2_config`. A new step is one function in the level's directory, one config field and one `if` block in that `calibrate`.

## L1: radiance

Table 1: L1 presets.

| Preset | Fields | Purpose |
|---|---|---|
| `iirspy` (default) | `empirical=True`, `interp_bands="linear"`, `interp_spatial=True`, `fix_frames=True`, `fill_frames=True` | Per-scene empirical dark and flat, with the LUT giving only the per-band scale; band and single-column gaps filled; band groups re-timed onto true frames and lost frames rebuilt, see [Frame defects](frames.md). |
| `issdc` | defaults | The per-element LUT gain and offset. |

The steps run in this order: residual dark, flat and smile, radiometry (`calib` picks the gain and offset, see `iirspy.l1.radiometry`), re-time frames, mask, fill (lost frames, then band and single-column gaps).

Per-scene inputs are keyword arguments, not config fields: `empirical_kws` (for example user-picked dark rows), `calib_dir`, `output_bands` and `frame_offsets` (the scene's frame table).

## L2: reflectance

The steps run in this order. Band numbers are IIRS band numbers, 1-indexed.

0. ISSDC polish (`polish="issdc"` only, before thermal removal as in SIS Appendix L): in I/F, multiply by ISSDC's gain per band and column (`issdc_gsp.csv`), mirrored when longitude falls along the columns since ISSDC indexes it in its east-right image; then, with d_b = R_b − (R_b−1 + R_b+1)/2, replace R_b by (R_b−1 + R_b+1)/2 where |d_b| > 0.9 · rms(d_b−1, d_b, d_b+1). Needs `geom`, all 256 bands and all 250 columns.
1. Thermal removal: `thermal`, `thermal_eps` (emissivity) and `thermal_bands`, the inclusive `(first, last)` band range whose mean brightness temperature is the pixel temperature. `thermal_eps * B(T)` is subtracted at every band, in radiance, so it precedes the flux division.
2. Solar flux: `flux` selects the solar spectrum and `sdist` scales it by the squared Sun distance.
3. Photometric normalization to the standard geometry i = 30°, e = 0°, g = 30°: `photom` is the model r, and reflectance = I/F · r(30, 0, 30) / r(i, e, g). A model is a `photometry.MODELS` name (`none`, `lambert`, `lommel_seeliger`, `lunar_lambert`, `besse_m3_highland`, `apparent`, `issdc`) or any callable of `cos_inc`, `cos_view`, `g` and `wl`, passed by name; one that also takes `iof` returns the reflectance itself (Hapke-type models; `apparent`, I/F / cos_inc as in SIS Appendix K; and `issdc`, 0.9018 × apparent, as in ISSDC's NDI; the factor is not in the SIS). `besse_m3_highland` is fitted to M3 highlands, for comparison with M3, not an IIRS model. `use_slope_az` applies the geometry's facet terms when it has topo; `cos_min` floors both cosines; `min_lit` nulls pixels less lit than it.
4. Polish: a per-band `polish` (anything but `"issdc"`) divides each band by a per-band factor, mapped by band number. Bands absent from the table get 1.0.
5. Spectral smoothing: `smoothing` is the moving-average width in bands and `smoothing_edges` sets the end bands. `"pair"` (width 3 only) sets the first band to the mean of bands 2–3 and the last to the mean of the last two.
6. Clip: `clip_negative` sets negative reflectance to zero. It runs last, after smoothing, as range validation does in the ISSDC product. Past about 4.4 µm, where the signal is almost all thermal, this zeroes about half of the pixels of an ISSDC NDI band.
7. Fill: `fill_nan` writes a value over every NaN (null pixels, lost frames, dropped bands) as the very last step. The `issdc` presets use 0, as ISSDC's NDI marks all its nulls with 0; the `iirspy` preset keeps NaN.

`--fix-frames` and `--fill-frames` (true/false, bare = true; `L1Config` fields) re-time the cube with the strip's frame table and rebuild the lost frames before L2, on either L1 route (see `frames.md`). Unset keeps the L1 preset: on for the `iirspy` L1, off for `issdc-nci`.

Table 2: L2 presets. Unlisted fields keep the `L2Config` defaults.

| Preset | Non-default fields | Purpose |
|---|---|---|
| `iirspy` (default) | `thermal="isothermal"` | Sun distance from SPICE, isothermal thermal removal at emissivity 1 over bands 227–247, Lambert model at the standard geometry, terrain used when supplied. |
| `issdc` | `sdist="none"`, `photom="issdc"`, `thermal="isothermal"`, `clip_negative=True`, `fill_nan=0`, `polish="issdc"`, `use_slope_az=False` | Reproduces the ISSDC L2 reflectance product (NDI). |
| `verma2022` | `sdist="none"`, `photom="apparent"`, `thermal="isothermal"`, `thermal_eps=0.95`, `thermal_bands=(225, 246)`, `clip_negative=True`, `polish="verma2022"`, `use_slope_az=False`, `smoothing=3` | The Verma et al. (2022) plugin's thermal correction, polish and smoothing. |

### Geometry input

`L1.calibrate(geom=...)` takes the scene's `Geometry`, the one output of every georef method (`L1.georef(method, ...)`, `method` a key of `iirspy.georef.GEOREF`). With topo and `use_slope_az` the photometric step uses its facet slope, aspect, lit fraction and sun; without topo the incidence is its per-pixel sun zenith; with no `geom` it is the cube's per-line `solar_inc` from the SPM file.

## Recorded metadata

- `l1.img.attrs["l1_config"]` and `l2.img.attrs["l2_config"]` are JSON strings of every field. `L2Config(**json.loads(l2.img.attrs["l2_config"]))` rebuilds the config. A callable `photom` is recorded by function name only.
- `l2.img.attrs` also holds the per-scene `L2.DIAGNOSTICS`: `solar_distance_au`, `inc_min_deg`, `inc_max_deg`, `inc_mean_deg`, `cos_inc_floor_frac`, `cos_view_floor_frac`, and `slope_az_used`.
- `attrs["provenance"]` is the JSON list of step entries from the raw cube on.
- Coordinate `T` [K] is the per-pixel temperature when thermal removal is on.

## Data files

Table 3: Data files read by the L2 steps, under `iirspy/data/`.

| File | Used by | Content |
|---|---|---|
| `spectral_polish_verma2022.csv` | `polish="verma2022"` | Per-band polish, bands 8–254. |
| `issdc_gsp.csv` | `polish="issdc"` | ISSDC's polish gain, band 1–256 by column `c000`–`c249` of the NDI (north-up, east-right) image, measured as NDI ard radiance / NCI radiance over 13 scenes. |
| `solar_flux_noaa_cdr_v03.csv` | `flux="noaa_cdr_mean"`, `"noaa_cdr_monthly"` | `mean_2010_2019` and one `YYYY-MM` column per month, mW/cm²/µm at 1 AU, convolved to the IIRS bands. The monthly lookup uses the scene label's start time. |

## References

- Besse, S., et al. (2013). A visible and near-infrared photometric correction for Moon Mineralogy Mapper (M3). Icarus, 222, 229–242. https://doi.org/10.1016/j.icarus.2012.10.036
- Coddington, O., Lean, J. L., Lindholm, C., Pilewskie, P. (2024). NOAA Climate Data Record (CDR) of NASA NOAA LASP Spectral Solar Irradiance (NNLSSI), Version 3. Monthly SSI, 2010-01 to 2025-12. NOAA National Centers for Environmental Information. https://doi.org/10.25921/esjz-1w61. Accessed 2026-10-09.
- Verma, Chauhan & Chauhan (2022). Lunar surface temperature estimation and thermal emission correction using Chandrayaan-2 imaging infrared spectrometer data for H2O & OH detection using 3 μm hydration feature. Icarus, 383, 115075. https://doi.org/10.1016/j.icarus.2022.115075
