"""
Forward model: modelled ¹⁰Be concentration at each profile depth.

The forward model integrates production over the sample thickness and
accumulates contributions from five production pathways:
    1  spallation (neutron)
    2  fast muon pathway 1
    3  fast muon pathway 2
    4  negative muon pathway 1
    5  negative muon pathway 2
    6  negative muon pathway 3

Direct translation of the core concentration loop in be_maincalc.m.
"""
import numpy as np


def sample_concentration(depth, thickness, density, v1, v2,
                          time, decay_const, erosion_rate, inheritance):
    """
    Modelled ¹⁰Be concentration for a single depth sample.

    Parameters
    ----------
    depth : float, top depth of sample (cm)
    thickness : float, full sample thickness (cm)
    density : float, bulk density at this depth (g/cm³)
    v1 : array (6,), effective attenuation lengths (g/cm²) for each pathway
    v2 : array (6,), surface production rates (atoms/g/yr) for each pathway
    time : float, exposure age (yr)
    decay_const : float, radioactive decay constant (yr⁻¹)
    erosion_rate : float, erosion rate (cm/yr)
    inheritance : float, pre-exposure ¹⁰Be concentration (atoms/g)

    Returns
    -------
    float, modelled concentration (atoms/g)
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    tmp1 = np.exp(-depth * density / v1)
    tmp2 = np.exp(-(depth + thickness) * density / v1)
    simp = decay_const + erosion_rate * density / v1
    tmp3 = (v1 * v2) * (np.exp(-time * simp) - 1.0) / (density * simp)

    return np.sum((tmp2 - tmp1) * tmp3) / thickness + inheritance * np.exp(-time * decay_const)


def profile_concentration(profile_data, v1, v2, time, decay_const,
                           erosion_rate, inheritance, densities):
    """
    Modelled ¹⁰Be concentrations for an entire depth profile.

    Parameters
    ----------
    profile_data : dict from io.read_profile_data()
    v1 : array (6,), effective attenuation lengths
    v2 : array (6,), surface production rates
    time : float, exposure age (yr)
    decay_const : float, decay constant (yr⁻¹)
    erosion_rate : float, erosion rate (cm/yr)
    inheritance : float, inheritance (atoms/g)
    densities : array (n_depths,), bulk density per sample (g/cm³)

    Returns
    -------
    array (n_depths,), modelled concentrations (atoms/g)
    """
    depths = profile_data["depth"]
    thicknesses = profile_data["thickness"]
    n = len(depths)
    modelled = np.zeros(n)
    for k in range(n):
        modelled[k] = sample_concentration(
            depths[k], thicknesses[k], densities[k],
            v1, v2, time, decay_const, erosion_rate, inheritance,
        )
    return modelled


def chi2_profile(modelled, measured, rel_error, dof):
    """
    Reduced chi-squared statistic for the profile.

    chi2 = sum[ ((N_mod - N_meas) / (N_meas * sigma_rel))^2 ] / dof

    Parameters
    ----------
    modelled : array, modelled concentrations
    measured : array, measured concentrations
    rel_error : array, relative 1σ errors (fractions)
    dof : int or float, degrees of freedom

    Returns
    -------
    float, reduced chi-squared
    """
    return float(np.sum(((modelled - measured) / (measured * rel_error)) ** 2) / dof)
