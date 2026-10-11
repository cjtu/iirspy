# Calibration configs

`L0.calibrate` converts raw counts to L1 radiance and `L1.calibrate` converts L1 radiance to L2 reflectance (I/F). Each method choice is one field of a frozen config, `iirspy.config.L1Config` or `iirspy.config.L2Config`. Named presets bundle the fields, and keyword overrides replace single fields.

```python
import iirspy

l0 = iirspy.L0("20210720T2333026105", "/path/to/data/")
l1 = l0.calibrate()                              # L1Config preset "iirspy"
l2 = l1.calibrate()                              # L2Config preset "iirspy"
l2 = l1.calibrate("issdc")                       # reproduce the ISSDC L2 product
l2 = l1.calibrate("issdc", sdist="spice")        # ...with one field overridden
```

## L1: radiance

Table 1: L1 presets.

| Preset | Fields | Purpose |
|---|---|---|
| `iirspy` (default) | `empirical=True`, `interp_bands="linear"`, `interp_spatial=True`, `fix_frames=True`, `fill_frames=True` | Per-scene empirical dark and flat, with the LUT giving only the per-band scale; band and single-column gaps filled; band groups re-timed onto true frames and lost frames rebuilt, see [Frame defects](frames.md). |
| `issdc` | defaults | The per-element LUT gain and offset. |

Per-scene inputs are keyword arguments, not config fields: `empirical_kws` (for example user-picked dark rows), `calib_dir`, `output_bands` and `frame_offsets` (the scene's frame table).

## L2: reflectance

The steps run in this order. Band numbers are IIRS band numbers, 1-indexed.

1. Solar flux: `flux` selects the solar spectrum and `sdist` scales it by the squared Sun distance.
2. Thermal removal: `thermal`, `thermal_eps` (emissivity) and `thermal_bands`, the inclusive `(first, last)` band range whose mean brightness temperature is the pixel temperature. `thermal_eps * B(T)` is subtracted at every band.
3. Clip: `clip_negative` sets negative thermal-removed radiance to zero.
4. Polish: `polish` divides each band by a per-band factor, mapped by band number. Bands absent from the table get 1.0.
5. Photometric normalization: `photom` is the disk function f(cos_inc, cos_view, g); `use_slope_az` applies a supplied terrain product; `cos_min` floors both cosines; `min_lit` nulls pixels less lit than it; `phase` rescales onto the Besse et al. (2013) 30° phase reference.
6. Spectral smoothing: `smoothing` is the moving-average width in bands and `smoothing_edges` sets the end bands. `"pair"` (width 3 only) sets the first band to the mean of bands 2–3 and the last to the mean of the last two.

Table 2: L2 presets. Unlisted fields keep the `L2Config` defaults.

| Preset | Non-default fields | Purpose |
|---|---|---|
| `iirspy` (default) | `thermal="isothermal"` | Sun distance from SPICE, isothermal thermal removal at emissivity 1 over bands 227–247, Lambert disk function, terrain used when supplied. |
| `issdc` | `sdist="none"`, `thermal="isothermal"`, `clip_negative=True`, `polish="issdc_empirical"`, `use_slope_az=False`, `smoothing=3` | Reproduces the ISSDC PRADAN L2 reflectance product. |
| `verma2022` | as `issdc`, but `thermal_eps=0.95`, `polish="verma2022"` | The Verma et al. (2022) thermal correction and polish. |

The spectral polish of the `issdc` preset was inferred empirically from published ISSDC L2 products. On 9 windows from 3 scenes, `issdc` matches them to within 0.2% median over 750–2300 nm.

### Per-scene inputs

These are keyword arguments of `L1.calibrate`, not config fields, so they are not recorded in `l2_config`:

- `inc`: solar incidence [deg], per line or per pixel. Default: the SPM file's per-line value.
- `slope_az`: camera-space slope and aspect [deg], optionally with a lit fraction, on the cube's grid. It is a raster, a `(slope, aspect[, lit])` tuple or a Dataset (`photometry.load_slope_az`). Make one from a DEM already on the camera grid with `photometry.slope_az_from_dem`.
- `sun_az_offset`: degrees added to the SPM solar azimuth to reach the grid-north frame of `slope_az`.

```python
from iirspy import photometry
from iirspy.config import L2Config

slope, aspect = photometry.slope_az_from_dem(dem)
l2 = l1.calibrate("iirspy", photom="lunar_lambert", cos_min=0.0872, slope_az=(slope, aspect))
l2 = l1.calibrate(L2Config.preset("issdc", use_slope_az=True), slope_az=(slope, aspect))
```

## Recorded metadata

- `l1.img.attrs["l1_config"]` and `l2.img.attrs["l2_config"]` are JSON strings of every field. `L2Config(**json.loads(l2.img.attrs["l2_config"]))` rebuilds the config. A callable `photom` is recorded by function name only.
- `l2.img.attrs` also holds the per-scene `L2.DIAGNOSTICS`: `solar_distance_au`, `inc_min_deg`, `inc_max_deg`, `inc_mean_deg`, `cos_inc_floor_frac`, `cos_view_floor_frac`, `slope_az_used` and `sun_az_offset`.
- Coordinate `T` [K] is the per-pixel temperature when thermal removal is on.

## Data files

Table 3: Data files read by the L2 steps, under `iirspy/data/`.

| File | Used by | Content |
|---|---|---|
| `spectral_polish_verma2022.csv` | `polish="verma2022"` | Per-band polish, bands 8–254. |
| `spectral_polish_issdc_empirical.csv` | `polish="issdc_empirical"` | Per-band polish inferred from ISSDC L2 products; `ci95_lo`, `ci95_hi`, `n_scenes` give the fit's spread. |
| `solar_flux_noaa_cdr_v03.csv` | `flux="noaa_cdr_mean"`, `"noaa_cdr_monthly"` | `mean_2010_2019` and one `YYYY-MM` column per month, mW/cm²/µm at 1 AU, convolved to the IIRS bands. The monthly lookup uses the scene label's start time. |

## References

- Besse, S., et al. (2013). A visible and near-infrared photometric correction for Moon Mineralogy Mapper (M3). Icarus, 222, 229–242. https://doi.org/10.1016/j.icarus.2012.10.036
- Coddington, O., Lean, J. L., Lindholm, C., Pilewskie, P. (2024). NOAA Climate Data Record (CDR) of NASA NOAA LASP Spectral Solar Irradiance (NNLSSI), Version 3. Monthly SSI, 2010-01 to 2025-12. NOAA National Centers for Environmental Information. https://doi.org/10.25921/esjz-1w61. Accessed 2026-10-09.
- Verma, Chauhan & Chauhan (2022). Lunar surface temperature estimation and thermal emission correction using Chandrayaan-2 imaging infrared spectrometer data for H2O & OH detection using 3 μm hydration feature. Icarus, 383, 115075. https://doi.org/10.1016/j.icarus.2022.115075
