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
| `iirspy` (default) | `dark="issdc_lut_median"`, `flat="flatfield_v1"`, `interp_bands="linear"`, `interp_spatial=True`, `fix_frame_defects=True`, `fill_frame_defects=True` | The packaged LUT's band-median offset and gain as the dark and scale, and the static quarterly flat `flatfield_v1`, which was derived on that setup; band and single-column gaps filled; band groups re-timed onto true frames and lost frames rebuilt, see [Frame defects](frame_defects.md). |
| `issdc` | `calib="issdc_derived"`, `dark="issdc_derived"`, `flat="issdc_derived"`, `keystone="simple"`, `edge_copy=True` | Reproduces ISSDC's NCI L1 from raw counts to a median of 0.2-0.7 %: the gain and offset table fitted to NCI, then its cross-track keystone resample and edge-column copy. |
| `lut-only` | `dark="issdc_lut_median"`, `flat=None` | The packaged LUT's band-median gain and offset; no flat. |

The steps run in this order: empirical frames (when asked for), dark subtraction (`dark`), flat division (`flat`, with the empirical smile), radiometry (`calib` picks the band scale), keystone (`keystone`: `None`, `"simple"` or `"weighted"`, which leaves flagged elements out of the interpolation; `edge_copy` is separate), re-time frame defects, mask, fill (frame defects, then band and single-column gaps).

Dark sources are `"issdc_lut"` (the whole packaged LUT, one offset per detector element), `"issdc_lut_median"` (its band-median offset), `"issdc_derived"` (fitted to ISSDC's NCI, `data/issdc_derived_*_e1g2.csv`) and `"empirical"` (the scene's shadow rows). Flat sources are `"issdc_lut"`, `"issdc_derived"`, `"flatfield_v1"` (`data/flatfield_v1.zip`, a quarterly static flat, one CSV per quarter) and `"empirical"` (the scene's own flat and smile). Both are applied to DN, before the scale. Files under `data/iir/` are ISSDC's own; everything we derived lives in `data/`. The keystone has two methods: `"simple"` is ISSDC's bicubic resample, `"weighted"` leaves flagged elements out of it so a bad column does not spread into its neighbours.

Per-scene inputs are keyword arguments, not config fields: `empirical_kws` (for example user-picked dark rows), `calib_dir`, `output_bands` and `frame_defects` (the scene's frame-defect table).

## L2: reflectance

The steps run in this order. Band numbers are IIRS band numbers, 1-indexed.

0. ISSDC polish (`polish="issdc"` only, before thermal removal as in SIS Appendix L): in I/F, multiply by ISSDC's gain per band and column (`issdc_gsp.csv`), mirrored when longitude falls along the columns since ISSDC indexes it in its east-right image; then, with d_b = R_b − (R_b−1 + R_b+1)/2, replace R_b by (R_b−1 + R_b+1)/2 where |d_b| > 0.9 · rms(d_b−1, d_b, d_b+1). Needs `geom`, all 256 bands and all 250 columns.
1. Thermal removal: `thermal`, `thermal_eps` (emissivity) and `thermal_bands`, the inclusive `(first, last)` band range that sets the pixel temperature (`isothermal`: mean brightness temperature; `issdc`: the T whose Planck radiance has the same band mean as the data). `thermal_eps * B(T)` is subtracted at every band, in radiance, so it precedes the flux division.
2. Solar flux: `flux` selects the solar spectrum and `sdist` scales it by the squared Sun distance.
3. Photometric normalization to the standard geometry i = 30°, e = 0°, g = 30°: `photom` is the model r, and reflectance = I/F · r(30, 0, 30) / r(i, e, g). A model is a `photometry.MODELS` name (`none`, `lambert`, `lommel_seeliger`, `lunar_lambert`, `besse_m3_highland`, `apparent`, `issdc`) or any callable of `cos_inc`, `cos_view`, `g` and `wl`, passed by name; one that also takes `iof` returns the reflectance itself (Hapke-type models; `apparent`, I/F / cos_inc as in SIS Appendix K; and `issdc`, 0.9018 × apparent, as in ISSDC's NDI; the factor is not in the SIS). `besse_m3_highland` is fitted to M3 highlands, for comparison with M3, not an IIRS model. `use_slope_az` applies the geometry's facet terms when it has topo; `cos_min` floors both cosines; `min_lit` nulls pixels less lit than it.
4. Polish: a per-band `polish` (anything but `"issdc"`) divides each band by a per-band factor, mapped by band number. Bands absent from the table get 1.0.
5. Spectral smoothing: `smoothing` is the moving-average width in bands and `smoothing_edges` sets the end bands. `"pair"` (width 3 only) sets the first band to the mean of bands 2–3 and the last to the mean of the last two.
6. Clip: `clip_negative` sets negative reflectance to zero. It runs last, after smoothing, as range validation does in the ISSDC product. Past about 4.4 µm, where the signal is almost all thermal, this zeroes about half of the pixels of an ISSDC NDI band.
7. Fill: `fill_nan` writes a value over every NaN (null pixels, lost frames, dropped bands) as the very last step. The `issdc` presets use 0, as ISSDC's NDI marks all its nulls with 0; the `iirspy` preset keeps NaN.

`--fix-frame-defects` and `--fill-frame-defects` (true/false, bare = true; `L1Config` fields) re-time the cube with the strip's frame-defect table and rebuild the lost frames before L2, on either L1 route (see `frame_defects.md`). Unset keeps the L1 preset: on for the `iirspy` L1, off for `issdc-nci`.

Table 2: L2 presets. Unlisted fields keep the `L2Config` defaults.

| Preset | Non-default fields | Purpose |
|---|---|---|
| `iirspy` (default) | `thermal="isothermal"` | Sun distance from SPICE, isothermal thermal removal at emissivity 1 over bands 227–247, Lambert model at the standard geometry, terrain used when supplied. |
| `issdc` | `sdist="none"`, `photom="issdc"`, `thermal="issdc"`, `thermal_bands=(226, 244)`, `clip_negative=True`, `fill_nan=0`, `polish="issdc"`, `use_slope_az=False` | Reproduces the ISSDC L2 reflectance product (NDI) from their L1 (NCI). |
| `verma2022` | `sdist="none"`, `photom="apparent"`, `thermal="isothermal"`, `thermal_eps=0.95`, `thermal_bands=(225, 246)`, `clip_negative=True`, `polish="verma2022"`, `use_slope_az=False`, `smoothing=3` | The Verma et al. (2022) plugin's thermal correction, polish and smoothing. |

How close the `issdc` preset gets, NCI -> NDI: 99% of pixels within 0.03% below 1.7 µm (the rest mostly where the spike rule fires differently, often in the edge columns); within about 0.3% at 1.7–2.5 µm; past 2.5 µm within a few percent, set by about 0.25 K of noise in ISSDC's temperature map that their delivered products do not carry.

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
