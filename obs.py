import os
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy import interpolate

def spec(
        path_spec, 
        path_telluric, 
        band, 
        ord_list, 
        ord_norm, 
        norm=True, 
        CH4mask=False, 
        lowermask=True,
        airmass_ratio=None,
        ):
    """Read the observed spectrum and normalize it.

    Args:
        path_spec (str): Path to the observed spectrum.
        path_telluric (str): Path to the telluric spectrum.
        band (str): Band of the observed spectrum.
        ord_list (list): List of orders to be used.
        ord_norm (int): Order to be used for normalization.
        norm (bool): If True, normalize the observed spectrum.
        CH4mask (bool): If True, mask the CH4 absorption lines.
        lowermask (bool): If True, mask the lower outliers.
    
    Returns:
        wavelength, flux, error, orders, normalization wavelength,
    """
    if CH4mask:
        read_args = {'header':None, 'names':['wav', 'order', 'flux', 'uncertainty']}
    else:
        read_args = {'header':None, 'sep':'\s+', 'names':['wav', 'order', 'flux', 'uncertainty']}
    
    dat = pd.read_csv(path_spec, **read_args)
    ld_obs, ord, f_obs_lmd, f_obserr_lmd = dat['wav'], dat['order'], dat['flux'], dat['uncertainty']
    if airmass_ratio is not None:
        f_obs_lmd = f_obs_lmd ** airmass_ratio
        f_obserr_lmd = (airmass_ratio * f_obs_lmd ** (airmass_ratio - 1)) * f_obserr_lmd
    if norm:
        wav_sort, flux_sort, flux_med, nflux, nflux_err = spec_norm(path_spec)
        f_obs_lmd = nflux
        f_obserr_lmd = nflux_err
        wav_sort = wav_sort * 1.0e1
    # [nm] => [A]
    ld_obs = ld_obs * 1.0e1
    if band=='h':
        ord = ord + 51

    mask = (ord==int(ord_norm))
    ld0 = np.nanmedian(ld_obs[mask])
    if not CH4mask:
        # f_obs_lmd => f_obs_nu
        f_obs_nu = f_obs_lmd * (ld_obs)**2.0e0
        f_obserr_nu = f_obserr_lmd * (ld_obs)**2.0e0

        # normalize by the median wavelength of ord_norm
        f_itp = interpolate.interp1d(ld_obs, f_obs_nu, kind='linear')
        f_obs0 = np.nanmedian(f_itp(ld_obs[mask])[(ld0-15<ld_obs[mask]) & (ld_obs[mask]<ld0+15)])
        ##norm=np.percentile(flux,90)#1#40000

        # normalize by the flux at ld0
        f_obs_nu = f_obs_nu / f_obs0
        f_obserr_nu = f_obserr_nu / f_obs0
    else:
        f_obs_nu = f_obs_lmd 
        f_obserr_nu = f_obserr_lmd

    ## Mask Outliers
    ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l = mask_outliers(
        path_telluric, ord, ord_list, dat, 
        ld_obs, f_obs_nu, f_obserr_nu, 
        CH4mask=CH4mask, lowermask=lowermask)


    if norm:
        return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, ld0, wav_sort, flux_med
    else:
        return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, ld0
    
def spec_norm(path_spec):
    """Normalize the observed spectrum.
    
    Args:
        path_spec (str): Path to the observed spectrum.

    Returns:
        wav_sort, flux_sort, flux_med, nflux, nflux_err
    """
    dat = pd.read_csv(path_spec)

    wav_sort, flux_sort = [], []
    for order in dat['order'].unique():
        dat_ord = dat[dat['order']==order]
        wav_sort.extend(dat_ord['wav'][200:-300].values)
        flux_sort.extend(dat_ord['flux'][200:-300].values)
    wav_sort = np.array(wav_sort)
    flux_sort = np.array(flux_sort)

    flux_med = medfilt(flux_sort, kernel_size=999)

    nflux = []
    nflux_err = []
    for order in dat['order'].unique():
        dat_ord = dat[dat['order']==order]
        norm = np.interp(dat_ord['wav'],wav_sort,flux_med)
        nflux.extend(dat_ord['flux'].values/norm)
        nflux_err.extend(dat_ord['uncertainty'].values/norm)

    nflux = np.array(nflux)
    nflux_err = np.array(nflux_err)
    return wav_sort, flux_sort, flux_med, nflux, nflux_err

def mask_outliers(path_telluric, ord, ord_list, dat, ld_obs, f_obs_nu, f_obserr_nu, CH4mask=False, lowermask=True):
    """Mask the outliers in the observed spectrum.
    
    Args:
        path_telluric (str): Path to the telluric spectrum.
        ord (pd.Series): Order of the observed spectrum.
        ord_list (list): List of orders to be used.
        dat (pd.DataFrame): Observed spectrum.
        ld_obs (pd.Series): Wavelength of the observed spectrum.
        f_obs_nu (pd.Series): Flux of the observed spectrum.
        f_obserr_nu (pd.Series): Error of the observed spectrum.
        CH4mask (bool): If True, mask the CH4 absorption lines.
        lowermask (bool): If True, mask the lower outliers.

    Returns:
        ld_obs_l, f_obs_nu_l, f_obs
    """
    wav_telluric, flux_telluric = read_telluric(path_telluric,wavlim=None)

    ld_obs_l = []
    f_obs_nu_l = []
    f_obserr_nu_l = []
    ord_l = []
    for k in range(len(ord_list)):
        ld_obs_k = []
        f_obs_nu_k = []
        f_obserr_nu_k = []
        ord_k = []
        for j in range(len(ord_list[k])):
            mask = ord==ord_list[k][j]
            dat_mask = dat[mask]
            if not CH4mask:
                # exclude order edges
                if len(dat_mask)<250:
                    ind_str, ind_end = dat_mask.index[0], dat_mask.index[-1]
                elif ord_list[k][j]>51: ##h band
                    ind_str, ind_end = dat_mask.index[0]+108, dat_mask.index[-1]-104
                else:
                    ind_str, ind_end = dat_mask.index[0]+216, dat_mask.index[-1]-104
            else:
                ind_str,ind_end = dat_mask.index[0], dat_mask.index[-1]+1
            ld_obs_j = ld_obs[mask].loc[ind_str:ind_end].values
            f_obs_nu_j = f_obs_nu[mask].loc[ind_str:ind_end].values
            f_obserr_nu_j = f_obserr_nu[mask].loc[ind_str:ind_end].values
            ord_j = ord[mask].loc[ind_str:ind_end].values
            if not CH4mask:
                # sigma clip
                #mask_clip = sigma_clip(f_obs_nu_j,sigma_lower=5,sigma_upper=3,maxiters=1).mask
                #mask telluric
                flux_telluric_interp = np.interp(ld_obs_j, wav_telluric, flux_telluric)
                tel_ind = np.where(flux_telluric_interp<0.95)[0]

                # mask with large error
                yerrfit = np.poly1d(np.polyfit(ld_obs_j,f_obserr_nu_j,2))(ld_obs_j)
                mask_ind = np.argsort((f_obserr_nu_j-yerrfit))[-15:] #upper outliers
                if lowermask:
                    for ind_tmp in np.argsort((yerrfit-f_obserr_nu_j))[::-1]:
                        if len(mask_ind)>19:
                            break
                        elif (ind_tmp not in tel_ind) and (ind_tmp not in mask_ind):
                            mask_ind = np.append(mask_ind,ind_tmp)

                    err_sort = np.argsort(np.abs(f_obserr_nu_j-yerrfit)) #significant outlier not telluric
                    count=0
                    for ind in err_sort[::-1]:
                        if (ind not in mask_ind) and (ind not in tel_ind):
                            mask_ind = np.append(mask_ind,ind)
                            count+=1
                            if count>4:
                                break
                mask_err = np.zeros(len(ld_obs_j),dtype=bool)
                for ind in mask_ind:
                    mask_err[ind] = True
                #mask_err = mask_err | mask_tel
            else:
                mask_err=np.zeros(len(ld_obs_j),dtype=bool)
            ld_obs_k.extend(ld_obs_j[~mask_err])
            f_obs_nu_k.extend(f_obs_nu_j[~mask_err])
            f_obserr_nu_j = f_obserr_nu_j[~mask_err]
            f_obserr_nu_j[f_obserr_nu_j==0] = 10. 
            f_obserr_nu_k.extend(f_obserr_nu_j)
            ord_k.extend(ord_j[~mask_err])
        ld_obs_k = np.array(ld_obs_k)
        f_obs_nu_k = np.array(f_obs_nu_k)
        f_obserr_nu_k = np.array(f_obserr_nu_k)
        ord_k = np.array(ord_k)

        # wavelength descending order
        ld_obs_k = ld_obs_k[::-1]
        f_obs_nu_k = f_obs_nu_k[::-1]
        f_obserr_nu_k = f_obserr_nu_k[::-1]
        ord_k = ord_k[::-1]

        ld_obs_l.append(ld_obs_k)
        f_obs_nu_l.append(f_obs_nu_k)
        f_obserr_nu_l.append(f_obserr_nu_k)
        ord_l.append(ord_k)
    return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l

def read_telluric(path_telluric, wavlim=[14000,17500]):
    """Read the telluric spectrum.

    Args:
        path_telluric (str): Path to the telluric spectrum.
        wavlim (list): Wavelength range to be used.

    Returns:
        wav_telluric, flux_telluric
    """
    telluric = pd.read_csv(path_telluric, header=None, sep='\s+')
    wav_telluric = telluric[0].values*1.e1 ##[AA]
    flux_telluric = telluric[4].values
    if wavlim is not None:
        ind = (wavlim[0]<=wav_telluric) & (wav_telluric<=wavlim[1])
        wav_telluric = wav_telluric[ind][::-1]
        flux_telluric = flux_telluric[ind][::-1]
    return wav_telluric, flux_telluric

# photometry
# PHARO (Palomar) <-- not available?
# WFCAM (UKIDSS; UKIRT Deep Sky Survey) is referred, so we use WFCAM filter.
# c.f. https://sites.astro.caltech.edu/palomar/observer/200inchResources/sensitivities.html
def read_photometry_file(path_obs, band):
    """Read photometry file.

    Args:
        path_obs (str): Path to the observed data.
        band (str): photometric band.

    Returns:
        wl_min, wl_max, wl_ref, tr_ref, R_p, Rinst_p
    """
    if band=='y':
        file_filter = "UKIRT_WFCAM.J_filter.dat" ##Jband
        wl_cut = 0 ##CHECK!!
    elif band=='h':
        file_filter = "UKIRT_WFCAM.H_filter.dat"
    wl_cut = 0
    d = np.genfromtxt(os.path.join(path_obs, file_filter))
    wl_ref = d[:,0] ##AA
    tr_ref = d[:,1] ##1.

    wl_min = np.min(wl_ref)+wl_cut
    wl_max = np.max(wl_ref)-wl_cut
    dlmd = (wl_max - wl_min) / len(wl_ref)
    Rinst_p = 0.5 * (wl_min + wl_max) / dlmd
    #####
    # wl_min = np.min(1.0e8/np.concatenate(nusd))
    # wl_max = np.max(1.0e8/np.concatenate(nusd))
    # Rinst_p = 3257
    #####
    R_p = Rinst_p * 2.**5 # 10 x instrumental spectral resolution
    return wl_min, wl_max, wl_ref, tr_ref, R_p, Rinst_p

def barycentric_correction(ra_deg, dec_deg, mjd):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u

    subaru = EarthLocation.from_geodetic(lat=19.82555556*u.deg, lon=-155.47611111*u.deg, height=4139*u.m)
    sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    time = Time(mjd, format='mjd', scale='utc')
    barycorr = sc.radial_velocity_correction(obstime=time, location=subaru)
    barycorr_kms = barycorr.to(u.km/u.s)
    return barycorr_kms.value

if __name__ == "__main__":
    mjd = 59390.46496304 #59372.64163201 #
    # HR7672B
    #ra_deg = 301.0258333
    #dec_deg = 17.0702778
    # HR7672A
    ra_deg = 301.0259167
    dec_deg = 17.0701861
    barycorr = barycentric_correction(ra_deg, dec_deg, mjd)
    print(barycorr) #km/s