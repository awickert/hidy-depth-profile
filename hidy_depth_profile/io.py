"""
File I/O for profile data, shielding data, and density profiles.
"""
import numpy as np


def read_profile_data(filename):
    """
    Read a ¹⁰Be depth-profile data file.

    Expected columns (space-delimited):
        depth_cm  half_thickness_cm  N_10Be_atoms_g  relative_error_fraction

    Returns
    -------
    dict with keys:
        depth        : top depth of each sample (cm)
        thickness    : full thickness of each sample (cm) = 2 * half_thickness
        concentration: measured ¹⁰Be concentration (atoms/g)
        rel_error    : relative 1σ error (fraction, e.g. 0.03)
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != 4:
        raise ValueError(
            f"Profile file must have 4 columns, got {data.shape[1]}: {filename}"
        )
    return {
        "depth": data[:, 0],
        "thickness": 2.0 * data[:, 1],
        "concentration": data[:, 2],
        "rel_error": data[:, 3],
    }


def read_shielding_data(filename):
    """
    Read a topographic shielding file.

    Expected columns: azimuth_deg  dip_deg
    Returns array of shape (n, 2).
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != 2:
        raise ValueError(
            f"Shielding file must have 2 columns, got {data.shape[1]}: {filename}"
        )
    return data  # columns: [azimuth, dip_angle_above_horizon]


def read_density_data(filename):
    """
    Read a depth-variable bulk density file.

    Expected columns: depth_cm  density_g_cm3  error_g_cm3
    Returns dict with keys: depths, densities, errors.
    The first depth entry must be 0.
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 2:
        raise ValueError(
            f"Density file must have at least 2 columns, got {data.shape[1]}: {filename}"
        )
    depths = data[:, 0]
    densities = data[:, 1]
    errors = data[:, 2] if data.shape[1] > 2 else np.zeros_like(densities)
    if depths[0] != 0:
        raise ValueError("First entry in density depth column must be 0.")
    return {"depths": depths, "densities": densities, "errors": errors}


def _int_piecewise_linear(xi, yi, a, b):
    """
    Integral from a to b of a piecewise linear function defined by (xi, yi).

    Outside the range of xi, the function is constant (first/last value).
    yi may be a 2D array (n_functions × n_xi); result has shape (n_functions,).

    Direct Python translation of be_int_poly1.m.
    """
    xi = np.asarray(xi, dtype=float)
    yi = np.atleast_2d(np.asarray(yi, dtype=float))
    nxi = len(xi)

    def _find_ind(val):
        idx = np.searchsorted(xi, val, side='right')
        return int(idx)

    aind = _find_ind(a)
    bind = _find_ind(b)

    swap = bind < aind
    if swap:
        a, b = b, a
        aind, bind = bind, aind

    def _interp_y(ind, x):
        if ind == 0:
            return yi[:, 0]
        if ind >= nxi:
            return yi[:, nxi - 1]
        slope = (yi[:, ind] - yi[:, ind - 1]) / (xi[ind] - xi[ind - 1])
        return yi[:, ind - 1] + (x - xi[ind - 1]) * slope

    if aind == bind:
        ya = _interp_y(aind, a)
        yb = _interp_y(bind, b)
        res = 0.5 * (yb + ya) * (b - a)
    else:
        ya = _interp_y(aind, a)
        yb = _interp_y(bind, b)
        # trapezoid from a to xi[aind]
        res = (yi[:, aind] + ya) * (xi[aind] - a)
        # trapezoid from xi[bind-1] to b
        res = res + (yb + yi[:, bind - 1]) * (b - xi[bind - 1])
        # middle segments
        for k in range(1, bind - aind):
            cind = aind + k
            res = res + (yi[:, cind] + yi[:, cind - 1]) * (xi[cind] - xi[cind - 1])
        res = res * 0.5

    return -res if swap else res


def cumulative_bulk_density(density_data, sample_depths, sample_thickness, n_mc=10000):
    """
    Estimate mean and std of cumulative bulk density for each sample.

    Translation of be_cumbulkdensity.m.

    Parameters
    ----------
    density_data : dict from read_density_data()
    sample_depths : array of sample top depths (cm)
    sample_thickness : array of full sample thicknesses (cm)
    n_mc : int, MC draws for uncertainty estimation

    Returns
    -------
    mean_cbd : array, mean cumulative bulk density per sample (g/cm³)
    std_cbd  : array, std of cumulative bulk density per sample (g/cm³)
    """
    dens_depths = density_data["depths"]
    densities = density_data["densities"]
    errors = density_data["errors"]
    nd = len(densities)

    maxdepth = float(np.ceil(max(np.max(dens_depths), np.max(sample_depths + sample_thickness))))
    dens_depths_ext = np.append(dens_depths, maxdepth)
    layer_thickness = np.diff(dens_depths_ext)

    # MC density samples: (n_mc, nd)
    rng = np.random.default_rng()
    dens_samples = rng.standard_normal((n_mc, nd)) * errors + densities

    # cumulative mass at each layer boundary: (n_mc, nd)
    cum_density = np.cumsum(dens_samples * layer_thickness, axis=1)

    # prepend the surface density and convert to g/cm³ by dividing by depth
    # Shape: (n_mc, nd+1); col 0 = density at z=0 (constant), cols 1:= cumulative average
    xi = dens_depths_ext  # length nd+1
    yi_first = np.tile(densities[0], (n_mc, 1))  # (n_mc, 1)
    # cum_density[:,k] = integral from 0 to dens_depths_ext[k+1] of rho dz
    # average density from 0 to depth = cum_density / depth
    yi_rest = cum_density / dens_depths_ext[1:]  # (n_mc, nd)
    yi = np.hstack([yi_first, yi_rest])  # (n_mc, nd+1)

    n_samples = len(sample_depths)
    mean_cbd = np.zeros(n_samples)
    std_cbd = np.zeros(n_samples)

    for k in range(n_samples):
        z_top = sample_depths[k]
        z_bot = sample_depths[k] + sample_thickness[k]
        res = _int_piecewise_linear(xi, yi.T, z_top, z_bot) / sample_thickness[k]
        mean_cbd[k] = np.mean(res)
        std_cbd[k] = np.std(res)

    return mean_cbd, std_cbd
