# IIRSpy

Toolkit for working with ISRO's Chandrayaan-2 IIRS dataset. This is a work in progress - comes with no guarantees, is not yet tested, and is subject to change without notice!

## Installation

1. Fork / clone this repository.
2. Get uv package manager [installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)
3. From the main `iirspy` directory run `uv run make test`
    * This will make a new virtual environment and run the `iirspy` tests
    * You can now use this `venv` from an IDE or `uv run ipython` for an interactive prompt

## Worked example

Check out the fully worked examples notebooks [here](https://github.com/cjtu/iirspy/tree/main/iirspy/examples).

## Quickstart

1. Download IIRS data from the PRADAN / ISSDC system (requires an account).
2. Use `iirspy` to read data by giving the datetime basename and a local directory. Use `L0` for raw or `L1` for calibrated. The package unzips the downloaded files if necessary.

    ```python
    import iirspy
    extent = (5, 245, 4000, 8000)  # Optional (xmin, xmax, ymin, ymax), defaults to full image
    iirs_l0_raw = iirspy.L0("20210720T2333026105", "/path/to/data/", extent)
    iirs_l1_rad = iirspy.L1("20210720T2333026105", "/path/to/data/", extent)
    ```

3. (Optional): Check that files were downloaded and unzipped correctly. Useful when files are added or changed. Prints nothing if successful.

    ```python
    iirs_l0_raw.checksum()
    iirs_l1_raw.checksum()
    ```

4. Plot images. Images are handled with `xarray` and can be accessed directly with `L0.img` or `L1.img`.

    ```python
    ax = iirs_l0_raw.plot()  # Plot band 12 by default

    # Plot desired band from raw and calibrated image side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    iirs_l0_raw.plot(band=79, ax=axs[0])
    iirs_l1_rad.plot(band=79, ax=axs[1])

    # Plot the median spectrum of a subset of the image using xarray
    iirs_l0_raw.img.sel(y=slice(6000, 6250)).median(dims=('x', 'y')).plot()
    ```

5. Calibration (work in progress)

    ```python
    l1_radiance = iirs_l0_raw.calibrate()
    l2_reflectance = iirs_l1_rad.calibrate()
    ```

## Other features

The IIRS classes `L0` and `L1` provide access to image metadata.

```python
l0 = iirspy.L0("20210720T2333026105", "/path/to/data/")

# Print paths to the image files in this image bundle
print("Qub path", l0.qub)
print("Hdr path", l0.hdr)
print("Xml path", l0.xml)  
# ...etc for lbr, oat, oath, spm. L1 files also have csv and csv-xml

# The projection, start and stop time, exposure and gain are stored in metadata
print("Metadata", l0.metadata)

# Not all metadata is loaded in by default, but you can access other values in the XML file with .metaget (thanks to pdr!)
print("Solar inc (deg):", l0.metaget("isda:solar_incidence"))
```

## Works in progress

### Calibrate from L0 raw to L1 radiance

Note: Raw data from ISSDCC is already dark subtracted.

1. Remove bad pixels from gain and offset array 
2. Apply destriping.
3. (Not implemented) Postprocessing (keystone correction, radiance adjustment at order-sorting-filters and at sensor edges).
4. Write subset data to file.

### Georeferencing

Currently works decently for equatorial L1 data using supplied GCPs but fails near poles. Using the `ldem_80s_20m` from LOLA:

- ingest tie-points and warp
- auto control points / img to img matching

### Reflectance correction

1. Read IIRS L1 data from `.qub`
2. Convert radiance units to SI (x0.01) [1000 mW/cm^2/sr/um] -> [W/m^2/sr/um]
   - (Optional) Find bad bands (`iirs_bad_bands.csv`) for this img's exposure and gain (`.xml`) and set those bands to nodata
   - (Optional) Run the fourier destriper on the IIRS image to smooth artefacts
3. Read solar spectrum (`ch2_iirs_solar_flux.txt`) and scale by solar distance (spice) at image collection time
4. Get local solar incidence and azimuth for each line of the image from `.spm`
   - (Optional) Adjust solar incidence by dem
     - Resample the `ldem` to the resolution of the IIRS image
     - Get slope from gradient of `ldem`
     - Dot product between solar inc/az vector and surface normal vector
5. Do I/F refl correction (radiance / (cos_inc \* solar_flux)) on all pixels
   - (Optional): apply spectral polish (don't - makes the spectra worse)
   - (Optional): apply spectral smoothing (usually do the 3-band boxcar)
6. Write to geotif or envi img/hdr format
   - (Optional): Edit envi hdr / geotiff metadata to add wavelength band labels

