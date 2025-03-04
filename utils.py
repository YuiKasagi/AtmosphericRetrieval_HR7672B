import jax.numpy as jnp

def nu_shift(nu, dv):
    """Apply Doppler shift

    Args:
        nu (array): wavenumber grid
        dv (float): velocity shift [km/s]

    Returns:
        array: shifted wavenumber grid
    """
    c = 2.99792458e5 #[km/s]
    dnu = dv*nu/c
    return dnu

def powerlaw_temperature_ptop(pressure, logPtop, T0, alpha):
    """Power-law temperature-pressure profile
    
    Args:
        pressure (array): pressure [bar]
        logPtop (array): log10 of the pressure at the top of the atmosphere [bar]
        T0 (float): temperature at the top of the atmosphere [K]
        alpha (float): temperature-pressure gradient

    Returns:
        array: temperature [K]
    """
    Ptop = 10**(logPtop)
    return T0*(pressure/Ptop)**alpha

def scale_speckle(nusd, nu_grid_list, f_obs_A, scale_star, relRV, obs_grid=True):
    """Scaling speckle spectrum

    Args:
        nusd (array): wavenumber grid
        nu_grid_list(array): wavenumber grid
        scale_star (float): scaling factor for the star
        relRV (float): relative RV [km/s]
        obs_grid (bool): True if the observed grid is used

    Returns:
        list: scaled speckle spectrum
    """
    f_speckle = []
    for k in range(len(nusd)):
        dnu_star = nu_shift(nusd[k], relRV)
        if obs_grid:
            f_obs_A_shift = jnp.interp(nusd[k], nusd[k]+dnu_star, f_obs_A[k])
        else:
            f_obs_A_shift = jnp.interp(nu_grid_list[k], nusd[k]+dnu_star, f_obs_A[k])
        f_speckle.append(scale_star*f_obs_A_shift)
    return f_speckle