# IIRSpy

This is a work in progress. May be buggy and subject to change w/o notice!

Main code is in: [iirspy/utils.py](iirspy/utils.py).

## Current Workflow

Follow below steps to correct IIRS data in python.

## Download IIRS

- Uses the ISSDC/PRADAN command line downloader I wrote in python
  - requires your ISRO PRADAN system login
  - retries on disconnect
  - refreshes the login session on timeout
  - comes with checksums that we can check to ensure correctness (should add this to the package so we can re-download if the checksum fails)

## Georeference to LOLA DEM

Using the `ldem_80s_20m` from LOLA.

1. Read image with lat/lon extent in python and find nearest latitudes given in the geom `.csv` file
2. Crop to desired lines using `gdal`
3. Import into QGIS
4. Select tie-points from `ldem` to IIRS image
5. Warp DEM to image coords (bilinear interpolation, thin plate spline)

## Correct IIRS in python

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

## Georeference image

- Read reflectance image into QGIS
- Choose tie points (or use the previous ones for dem, reversed)
- Project to Spole stereo (`IAU_2015:30135`)
