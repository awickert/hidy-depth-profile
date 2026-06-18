"""
Cosmogenic nuclide production rate calculations.

Covers:
- Heisinger 2002 muon flux model (fast + negative muon pathways)
- Exponential fitting of muon production vs. depth curves
- Stone 2000 and LSDn spallation surface rates via the stoneage package
- Topographic + geometric shielding factor
"""
import sys
import numpy as np
from scipy import integrate, optimize

# stoneage lives as a sibling package
sys.path.insert(0, '/home/awickert/dataanalysis/stoneage')


# ---------------------------------------------------------------------------
# Heisinger 2002 subfunctions
# ---------------------------------------------------------------------------

def _Rv0(z):
    """Stopping rate of vertical muons at SLHL (muons/g/s)."""
    z = np.asarray(z, dtype=float)
    a = np.exp(-5.5e-6 * z)
    b = z + 21000.0
    c = (z + 1000.0) ** 1.66 + 1.567e5
    dadz = -5.5e-6 * a
    dbdz = np.ones_like(z)
    dcdz = 1.66 * (z + 1000.0) ** 0.66
    return -5.401e7 * (b * c * dadz - a * (c * dbdz + b * dcdz)) / (b ** 2 * c ** 2)


def _LZ(z):
    """Effective atmospheric attenuation length for muons of range z (g/cm²)."""
    # Groom et al. 2001 table (momentum MeV/c, range g/cm²) — restricted to ≤ 2e5 g/cm²
    _data = np.array([
        [4.704e1, 8.516e-1], [5.616e1, 1.542e0], [6.802e1, 2.866e0],
        [8.509e1, 5.698e0],  [1.003e2, 9.145e0],  [1.527e2, 2.676e1],
        [1.764e2, 3.696e1],  [2.218e2, 5.879e1],  [2.868e2, 9.332e1],
        [3.917e2, 1.524e2],  [4.945e2, 2.115e2],  [8.995e2, 4.418e2],
        [1.101e3, 5.534e2],  [1.502e3, 7.712e2],  [2.103e3, 1.088e3],
        [3.104e3, 1.599e3],  [4.104e3, 2.095e3],  [8.105e3, 3.998e3],
        [1.011e4, 4.920e3],  [1.411e4, 6.724e3],  [2.011e4, 9.360e3],
        [3.011e4, 1.362e4],  [4.011e4, 1.776e4],  [8.011e4, 3.343e4],
        [1.001e5, 4.084e4],  [1.401e5, 5.495e4],  [2.001e5, 7.459e4],
    ])
    log_range = np.log(_data[:, 1])
    log_p = np.log(_data[:, 0])
    z = np.asarray(z, dtype=float)
    z_safe = np.where(z < 1.0, 1.0, z)
    log_z = np.log(z_safe)
    log_z_clipped = np.clip(log_z, log_range[0], log_range[-1])
    P_MeVc = np.exp(np.interp(log_z_clipped, log_range, log_p))
    return 263.0 + 150.0 * (P_MeVc / 1000.0)


def muon_production_at_depth(massdepth, elev, isotope='Be-10'):
    """
    Compute fast and negative muon production rates at given mass depths.

    Direct translation of be_muonproduction.m (Heisinger 2002 model).

    Parameters
    ----------
    massdepth : array-like, g/cm²
    elev : float, elevation in metres above sea level
    isotope : str, 'Be-10' or 'Al-26'

    Returns
    -------
    total, fastmuons, negmuons : arrays of production rates (atoms/g/yr)
    """
    z = np.atleast_1d(np.asarray(massdepth, dtype=float))

    # site atmospheric pressure and atmospheric depth
    h = 1013.25 * np.exp((-0.03417 / 0.0065) * (np.log(288.15) - np.log(288.15 - 0.0065 * elev)))
    H = (1013.25 - h) * 1.019716

    a_c = 258.5 * (100.0 ** 2.66)
    b_c = 75.0 * (100.0 ** 1.66)
    phi_vert_slhl = (a_c / ((z + 21000.0) * ((z + 1000.0) ** 1.66 + b_c))) * np.exp(-5.5e-6 * z)

    R_vert_site = _Rv0(z) * np.exp(H / _LZ(z))

    # integrate Rv0(x)*exp(H/LZ(x)) from z_i to 2e5+1 for each depth
    phi_vert_site = np.zeros(len(z))
    for i, zi in enumerate(z):
        tol = max(phi_vert_slhl[i] * 1e-4, 1e-20)
        val, _ = integrate.quad(
            lambda x: _Rv0(np.array([x]))[0] * np.exp(H / _LZ(np.array([x]))[0]),
            zi, 2e5 + 1,
            limit=200, epsabs=tol, epsrel=1e-4,
        )
        phi_vert_site[i] = val

    # constant of integration: invariant flux at 2e5 g/cm²
    phi_200k = (a_c / ((2e5 + 21000.0) * ((2e5 + 1000.0) ** 1.66 + b_c))) * np.exp(-5.5e-6 * 2e5)
    phi_vert_site += phi_200k

    # angular distribution
    nofz = 3.21 - 0.297 * np.log((z + H) / 100.0 + 42.0) + 1.21e-5 * (z + H)
    dndz = (-0.297 / 100.0) / ((z + H) / 100.0 + 42.0) + 1.21e-5

    # total muon flux (muons/cm²/yr) and stopping rate (neg muons/g/yr)
    phi = phi_vert_site * 2.0 * np.pi / (nofz + 1.0) * 3.15576e7
    R_temp = (2.0 * np.pi / (nofz + 1.0)) * R_vert_site \
             - phi_vert_site * (-2.0 * np.pi * (nofz + 1.0) ** -2) * dndz
    R = R_temp * 0.44 * 3.15576e7

    # depth-dependent cross-section parameters
    Beta = 0.846 - 0.015 * np.log(z / 100.0 + 1.0) + 0.003139 * (np.log(z / 100.0 + 1.0)) ** 2
    Ebar = 7.6 + 321.7 * (1.0 - np.exp(-8.059e-6 * z)) + 50.7 * (1.0 - np.exp(-5.05e-7 * z))
    aalpha = 0.75

    if isotope == 'Be-10':
        sigma0 = (0.094e-27 / 1.096) / (190.0 ** aalpha)
        fastmuons = phi * Beta * (Ebar ** aalpha) * sigma0 * 2.006e22
        negmuons = R * (0.704 * 0.1828 * 0.0043) / 1.096
    else:  # Al-26
        sigma0 = (1.41e-27) / (190.0 ** aalpha)
        fastmuons = phi * Beta * (Ebar ** aalpha) * sigma0 * 1.003e22
        negmuons = R * (0.296 * 0.6559 * 0.022)

    return fastmuons + negmuons, fastmuons, negmuons


# ---------------------------------------------------------------------------
# Exponential curve fitting (translation of be_fitexp.m)
# ---------------------------------------------------------------------------

def _fit_sum_of_exp(n_exp, z, y):
    """
    Fit a sum of n_exp exponentials of the form sum_i a_i * exp(-z/b_i).

    Returns
    -------
    coeff : array [a1, ..., aN, b1, ..., bN]
    rel_err : mean relative error of the fit
    """
    _p0_defaults = {
        1: [0.05, 2000.0],
        2: [0.09, 0.02, 738.0, 2688.0],
        3: [0.09, 0.02, 0.02, 738.0, 2688.0, 4360.0],
    }
    p0 = _p0_defaults[n_exp]

    def model(z, *params):
        n = n_exp
        return sum(params[i] * np.exp(-z / params[n + i]) for i in range(n))

    # amplitudes ≥ 0, attenuation lengths > 0
    lower = [0.0] * n_exp + [1.0] * n_exp
    upper = [np.inf] * (2 * n_exp)

    popt, _ = optimize.curve_fit(
        model, z, y, p0=p0, bounds=(lower, upper),
        max_nfev=20000, method='trf',
    )
    residuals = y - model(z, *popt)
    rel_err = float(np.mean(np.abs(residuals / np.where(y == 0, 1e-30, y))))
    return popt, rel_err


def fit_muon_curves(elev, isotope='Be-10', fit_depth_m=2000):
    """
    Fit multi-exponential curves to the Heisinger muon production profiles.

    Translation of be_calcmuonproduction.m.

    Parameters
    ----------
    elev : float, elevation in metres
    isotope : str, 'Be-10' or 'Al-26'
    fit_depth_m : float, depth to which curves are fitted (metres)

    Returns
    -------
    dict with keys:
        fast_coeff   : [a1, a2, b1, b2] — fast muon exponential coefficients
        neg_coeff    : [a1, a2, a3, b1, b2, b3] — negative muon coefficients
        fast_surface : total fast muon surface rate (a1+a2, atoms/g/yr)
        neg_surface  : total negative muon surface rate (atoms/g/yr)
        total_muon   : fast_surface + neg_surface
        fast_relerr  : mean relative fitting error for fast muons
        neg_relerr   : mean relative fitting error for negative muons
    """
    max_massdepth = 2.7 * 100.0 * fit_depth_m  # g/cm²; 2.7 g/cm³ assumed rock
    massdepths = np.linspace(0.0, max_massdepth, 100)

    _, fastmuons, negmuons = muon_production_at_depth(massdepths, elev, isotope)

    fast_coeff, fast_relerr = _fit_sum_of_exp(2, massdepths, fastmuons)
    neg_coeff, neg_relerr = _fit_sum_of_exp(3, massdepths, negmuons)

    # amplitudes are the first n_exp elements
    fast_surface = float(np.sum(fast_coeff[:2]))
    neg_surface = float(np.sum(neg_coeff[:3]))

    return {
        "fast_coeff": fast_coeff,      # [a1, a2, b1, b2]
        "neg_coeff": neg_coeff,        # [a1, a2, a3, b1, b2, b3]
        "fast_surface": fast_surface,  # a1 + a2  (atoms/g/yr at surface)
        "neg_surface": neg_surface,    # a1+a2+a3 (atoms/g/yr at surface)
        "total_muon": fast_surface + neg_surface,
        "fast_relerr": fast_relerr,
        "neg_relerr": neg_relerr,
    }


# ---------------------------------------------------------------------------
# Spallation production rate
# ---------------------------------------------------------------------------

def stone2000_surface_rate(lat, lon, elev, reference_rate, isotope='Be-10'):
    """
    Stone 2000 spallation surface production rate at site using stoneage.

    Parameters
    ----------
    lat, lon : float, decimal degrees (south / west negative)
    elev : float, metres above sea level
    reference_rate : float, SLHL production rate (at/g/yr)
    isotope : str, 'Be-10' or 'Al-26'

    Returns
    -------
    float, surface spallation production rate at site (at/g/yr)
    """
    from stoneage.atmosphere import ERA40atm
    from stoneage.scaling import stone2000

    P_hPa = ERA40atm(lat, lon, elev)[0]
    SF_sp = stone2000(lat, P_hPa, Fsp=1.0)[0]
    rate = reference_rate * SF_sp

    if isotope == 'Al-26':
        rate *= 6.75  # Al-26 / Be-10 production ratio

    return float(rate)


def lsdn_surface_rate(lat, lon, elev, assumed_age_yr,
                      reference_rate, collection_year=2024, isotope='Be-10'):
    """
    Time-averaged LSDn spallation surface production rate at site using stoneage.

    Parameters
    ----------
    lat, lon, elev : site location
    assumed_age_yr : integration window (yr before collection)
    reference_rate : SLHL reference rate (at/g/yr)
    collection_year : CE year of sample collection
    isotope : str, 'Be-10' or 'Al-26'

    Returns
    -------
    float, time-averaged surface production rate (at/g/yr)
    """
    from stoneage.atmosphere import ERA40atm
    from stoneage.scaling import get_LSDnSF
    from stoneage.cutoff_rigidity import get_DipRc
    from stoneage.constants import make_consts

    consts = make_consts()
    nuclide = 'N10quartz' if isotope == 'Be-10' else 'N26quartz'
    nidx = consts.nuclides.index(nuclide)

    rin = {
        "lat": np.array([float(lat)]),
        "long": np.array([float(lon)]),
        "yr": np.array([float(collection_year)]),
        "t": np.array([float(assumed_age_yr)]),
    }
    sfdata = get_DipRc(rin)

    sfdata2 = {
        "tmax": [sfdata["tmax"][0]],
        "tmin": [sfdata["tmin"][0]],
        "Rc": [sfdata["Rc"][0]],
        "S": [sfdata["S"][0]],
        "nuclide": [nuclide],
        "pressure": np.array([ERA40atm(lat, lon, elev)[0]]),
    }
    sfdata2 = get_LSDnSF(sfdata2)

    dt = sfdata2["tmax"][0] - sfdata2["tmin"][0]
    mean_SF = np.average(sfdata2["LSDn"][0], weights=dt)

    return float(consts.refP_LSDn[nidx] * mean_SF)


# ---------------------------------------------------------------------------
# Topographic / geometric shielding factor
# ---------------------------------------------------------------------------

def shielding_factor(shielding_data, strike=0.0, dip=0.0):
    """
    Compute the combined topographic + geometric shielding factor.

    Translation of be_shielding_factor.m.

    Parameters
    ----------
    shielding_data : array (n, 2) with columns [azimuth_deg, dip_angle_deg]
    strike : float, strike of the surface (degrees)
    dip : float, dip of the surface (degrees)

    Returns
    -------
    float, shielding factor (0–1)
    """
    azimuth = shielding_data[:, 0]
    angle = shielding_data[:, 1]

    azimuth_interp = np.arange(360)
    # linearly interpolate (with extrapolation) the horizon angle
    angle_interp = np.interp(azimuth_interp, azimuth, angle,
                              left=angle[0], right=angle[-1])

    # geometric shielding due to surface dip
    dip_shielding = np.degrees(np.arctan(
        np.tan(np.radians(dip)) * np.cos(np.radians(azimuth_interp - strike + 90.0))
    ))

    total_shielding = np.maximum(angle_interp, dip_shielding)
    return float(np.mean(1.0 - np.sin(np.radians(total_shielding)) ** 3.3))
