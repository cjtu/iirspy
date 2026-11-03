import time
import warnings

import numpy as np

from iirspy import io


def flat_surface_temperature(inc, sdist=1.0, s0=1370.0, eps=0.95, a0=0.08, a=0.045, b=0.14):
    """Radiative-equilibrium dayside temperature [K] of a flat surface at solar incidence `inc` [deg].

    T = [S (1 - A) cos(i) / (eps sigma)]^(1/4), Williams et al. (2017) Eq. 1, with the incidence-dependent
    albedo A(i) = A0 + a (i/45)^3 + b (i/90)^8 of Vasavada et al. (2012); the defaults are Williams' fit to
    the Diviner zonal-mean noon temperatures (where i = latitude). S = `s0` / `sdist`^2. 0 K where i >= 90.

    Williams, J.-P., Paige, D. A., Greenhagen, B. T., & Sefton-Nash, E. (2017). The global surface temperatures
    of the Moon as measured by the Diviner Lunar Radiometer Experiment. Icarus, 283, 300-325.
    Vasavada, A. R., et al. (2012). Lunar equatorial surface temperatures and regolith properties from the
    Diviner Lunar Radiometer Experiment. JGR Planets, 117, E00H18.

    >>> tuple(round(float(flat_surface_temperature(i)), 1) for i in (0, 60, 90))
    (391.1, 318.4, 0.0)
    """
    albedo = a0 + a * (inc / 45) ** 3 + b * (inc / 90) ** 8
    cos_i = np.clip(np.cos(np.radians(inc)), 0, None)
    return (s0 / sdist**2 * (1 - albedo) * cos_i / (eps * 5.670374e-8)) ** 0.25


def thermal_radiance(rad, eps=1.0, bands=(227, 247), inc=None, sdist=1.0, t_min=0.0):
    """(eps * B(T), T) for an isothermal surface, T [K] per pixel.

    T is the mean brightness temperature at emissivity `eps` over IIRS band numbers `bands`
    (1-indexed, inclusive), selected by band number. `rad` is (band, y, x) [W/m^2/sr/um] with a
    `wl` [nm] coord. Where T is not above `t_min` (or not finite) and `inc` [deg] is given, T falls
    back to `flat_surface_temperature(inc, sdist)`. A cube with none of `bands` has no thermal
    term: (0.0, None).
    """
    tb = rad.sel(band=slice(*bands))
    if not tb.sizes["band"]:
        warnings.warn(f"no thermal bands {bands} in the cube, thermal removal skipped", UserWarning, stacklevel=2)
        return 0.0, None
    c, h, k = 3e8, 6.626e-34, 1.38e-23
    wl_t = tb.wl * 1e-9
    q = np.log(eps * 2 * h * c**2 * 1e-6 / (tb * wl_t**5) + 1)
    t = (h * c / (wl_t * k * q)).mean("band")
    if inc is not None:
        t = t.where(t > t_min, flat_surface_temperature(inc, sdist))
    wl = rad.wl * 1e-9
    bbr = 1e-6 * (2 * h * c * c / wl**5) * 1 / (np.exp(h * c / (wl * k * t)) - 1)
    return eps * bbr, t


def subtract(rad, cfg, *, ws=None, sdist, inc, **_):
    """Remove the isothermal thermal tail eps * B(T) from `rad`; the temperature [K] rides along as coord `T`."""
    if cfg.thermal == "none":
        return rad
    t0 = time.monotonic()
    trad, temp = thermal_radiance(
        rad, cfg.thermal_eps, cfg.thermal_bands, inc=inc, sdist=sdist, t_min=cfg.thermal_t_min
    )
    out = (rad - trad).assign_coords(T=temp)
    fields = ("thermal_eps", "thermal_bands", "thermal_t_min")
    return io.record_step(out, rad, "thermal", cfg.thermal, t0, cfg, fields)
