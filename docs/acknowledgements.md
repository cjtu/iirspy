# Acknowledgements

iirspy is released under the [MIT License](https://github.com/cjtu/iirspy/blob/main/LICENSE). It does not redistribute
any third-party code; the software below is installed by the user (or by pip) and used under its own license.

## AROSICS

This project uses **AROSICS** (Automated and Robust Open-Source Image Co-Registration Software) for image alignment,
developed by Helmholtz-Zentrum Potsdam - Deutsches GeoForschungsZentrum GFZ, licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

AROSICS is not a dependency in `pyproject.toml`. Install it into a separate environment with
[`arosics-environment.yml`](https://github.com/cjtu/iirspy/blob/main/arosics-environment.yml).

If you use iirspy's georeferencing, please cite:

> Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., & Hostert, P. (2017). *AROSICS: An Automated and Robust
> Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data*. Remote Sensing, 9(7), 676.
> [https://doi.org/10.3390/rs9070676](https://doi.org/10.3390/rs9070676)

## PDR

iirspy reads IIRS products with **PDR** (the Planetary Data Reader), developed by Million Concepts and licensed under the
[BSD 3-Clause License](https://github.com/MillionConcepts/pdr/blob/main/LICENSE). It is a regular dependency in
`pyproject.toml` and is installed from PyPI.

If you use iirspy for published work, please also cite:

> Brown, S. V., St. Clair, M., Million, C., Curtis, S., Aye, K.-M., & Weinberg, Z. (2024). *PDR: The Planetary Data
> Reader*. Journal of Open Source Software, 9(102), 7256.
> [https://doi.org/10.21105/joss.07256](https://doi.org/10.21105/joss.07256)

## Other dependencies

The remaining runtime dependencies are listed in
[`pyproject.toml`](https://github.com/cjtu/iirspy/blob/main/pyproject.toml) and are installed from PyPI. Each is
distributed under its own license; see the project pages for details.
