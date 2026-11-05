"""Thermal tail removal, the first L2 step: subtract the Moon's own emission before dividing by the solar flux.

Past ~2.5 um the dayside Moon emits as well as reflects. Each pixel is taken as isothermal: T is its mean brightness
temperature over `bands` (227-247, 4.5-4.9 um, where emission dominates), and eps * B(T, wl) is subtracted from
every band, in radiance:

    rad_reflected = rad - eps * B(T, wl)

Where that T is not above `t_min` (shadow, noise), it falls back to the radiative-equilibrium temperature of a flat
surface at the pixel's incidence (`flat_surface_temperature`): ~391 K at overhead sun, 0 K at the terminator.
"""

import warnings

import numpy as np

C = 3e8  # speed of light [m/s]
H = 6.626e-34  # Planck [J s]
K_B = 1.38e-23  # Boltzmann [J/K]
SIGMA = 5.670374e-8  # Stefan-Boltzmann [W/m^2/K^4]


def subtract(rad, eps=1.0, bands=(227, 247), inc=None, sdist=1.0, t_min=0.0, model="isothermal"):
    """`rad` less the isothermal tail eps * B(T), T [K] riding along as coord `T` (see `thermal_radiance`).

    model="issdc" takes T and B(T) from `issdc_thermal_radiance` instead (eps, inc, sdist, t_min unused).
    A cube with none of `bands` comes back unchanged, with a warning.
    """
    if model == "issdc":
        trad, temp = issdc_thermal_radiance(rad, bands)
    else:
        trad, temp = thermal_radiance(rad, eps, bands, inc=inc, sdist=sdist, t_min=t_min)
    return rad if temp is None else (rad - trad).assign_coords(T=temp)


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
    wl_t = tb.wl * 1e-9
    q = np.log(eps * 2 * H * C**2 * 1e-6 / (tb * wl_t**5) + 1)
    t = (H * C / (wl_t * K_B * q)).mean("band")
    if inc is not None:
        t = t.where(t > t_min, flat_surface_temperature(inc, sdist))
    wl = rad.wl * 1e-9
    bbr = 1e-6 * (2 * H * C * C / wl**5) * 1 / (np.exp(H * C / (wl * K_B * t)) - 1)
    return eps * bbr, t


def issdc_thermal_radiance(rad, bands=(226, 244)):
    """(B(T), T) as ISSDC's NDI: T solves mean(B(T, wl)) = mean(rad) over `bands`, emissivity 1.

    Uses ISSDC's constants h = 6.626e-34, c = 2.998e8, k = 1.38e-23. Matches the NDI `tem` map with no offset
    (median within 0.03 K on five scenes) up to a ~0.25 K pixel noise that no band of the radiance explains. Where
    the band-mean radiance is not positive, T is NaN and nothing is subtracted.
    """
    h, c, k = 6.626e-34, 2.998e8, 1.38e-23
    tb = rad.sel(band=slice(*bands))
    if not tb.sizes["band"]:
        warnings.warn(f"no thermal bands {bands} in the cube, thermal removal skipped", UserWarning, stacklevel=2)
        return 0.0, None
    wl = tb.wl * 1e-9
    a, x = 2 * h * c**2 * 1e-6 / wl**5, h * c / (wl * k)
    lm = tb.mean("band")
    wm = wl.mean()
    with np.errstate(invalid="ignore", over="ignore"):
        t = h * c / (wm * k * np.log(2 * h * c**2 * 1e-6 / (lm * wm**5) + 1))
        for _ in range(6):  # Newton from T_b at the mean wavelength
            ex = np.exp(x / t)
            b = a / (ex - 1)
            t = t + (lm - b.mean("band")) / (b * ex / (ex - 1) * x / t**2).mean("band")
        wl = rad.wl * 1e-9
        bbr = 2 * h * c**2 * 1e-6 / wl**5 / (np.exp(h * c / (wl * k * t)) - 1)
    return bbr.fillna(0.0), t


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
    return (s0 / sdist**2 * (1 - albedo) * cos_i / (eps * SIGMA)) ** 0.25
