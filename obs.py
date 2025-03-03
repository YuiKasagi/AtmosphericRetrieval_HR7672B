import numpy as np
import os
import pandas as pd
from scipy.signal import medfilt

def spec(path_obs, ord_list, ord_norm):
    d = np.genfromtxt(os.path.join(path_obs, "multiply_bb_20221221.dat"))
    d = d[np.argsort(d[:, 0])]
    # [um], f_lmd
    ld_obs, f_obs_lmd, f_obserr_lmd, ord = d[:,0], d[:,1], d[:, 2], d[:, 3]
    ord = np.array(ord, dtype='int')

    # [um] => [A]
    ld_obs = ld_obs * 1.0e4

    # f_obs_lmd => f_obs_nu
    f_obs_nu = f_obs_lmd * (ld_obs)**2.0e0
    f_obserr_nu = f_obserr_lmd * (ld_obs)**2.0e0


    # normalize by the median wavelength of ord_norm
    mask = (ord==int(ord_norm))
    ld0 = np.median(ld_obs[mask])
    f_itp = interpolate.interp1d(ld_obs, f_obs_nu, kind='linear')
    f_obs0 = f_itp(ld0)

    # normalize by the flux at ld0
    f_obs_nu = f_obs_nu / f_obs0
    f_obserr_nu = f_obserr_nu / f_obs0

    f_ref = norm_flux(path_obs, ld0)


    ld_obs_l = []
    f_obs_nu_l = []
    f_obserr_nu_l = []
    ord_l = []
    for k in range(len(ord_list)):
        # order masking
        mask = (ord>=ord_list[k][0]) * (ord<=ord_list[k][1])
        ld_obs_k = ld_obs[mask]
        f_obs_nu_k = f_obs_nu[mask]
        f_obserr_nu_k = f_obserr_nu[mask]
        ord_k = ord[mask]

        # wavelength descending order
        ld_obs_k = ld_obs_k[::-1]
        f_obs_nu_k = f_obs_nu_k[::-1]
        f_obserr_nu_k = f_obserr_nu_k[::-1]
        ord_k = ord_k[::-1]

        ld_obs_l.append(ld_obs_k)
        f_obs_nu_l.append(f_obs_nu_k)
        f_obserr_nu_l.append(f_obserr_nu_k)
        ord_l.append(ord_k)

    return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, f_ref



from scipy import interpolate
def norm_flux(path_obs, ld0):
    d = np.genfromtxt(os.path.join(path_obs, "T7_Gl229B.txt"))
    # [um], [W/m^2/um]
    ld_ref, f_ref = d[:,0], d[:,1]

    # [um] => [A]
    ld_ref = ld_ref * 1.0e4

    # [W/m^2/um] => [erg/s/cm^2/cm]
    f_ref = f_ref * 1.0e7 * 1.0e-4 * 1.0e4
    # [erg/s/cm^2/cm] => [erg/s/cm^2/cm^{-1}] # units of exojax
    f_ref = f_ref * (ld_ref * 1.0e-8)**2.0e0

    mask = (f_ref > 0.)
    f_itp = interpolate.interp1d(ld_ref[mask], f_ref[mask], kind='linear')
    f0 = f_itp(ld0) # ref flux at ld0

    return f0

from astropy.stats import sigma_clip
def spec_kasagi(path_spec, band, ord_list, ord_norm, norm=True, CH4mask=False, lowermask=True):
    if CH4mask:
        read_args={'header':None,'names':['wav','order','flux','uncertainty']}
    else:
        read_args={'header':None,'sep':'\s+','names':['wav','order','flux','uncertainty']}
    
    dat = pd.read_csv(path_spec,**read_args)
    ld_obs, ord, f_obs_lmd, f_obserr_lmd = dat['wav'], dat['order'], dat['flux'], dat['uncertainty']
    if norm:
        wav_sort, flux_sort, flux_med, nflux, nflux_err = spec_kasagi_norm(path_spec)
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
        #f_obs0 = f_itp(ld0)
        f_obs0 = np.nanmedian(f_itp(ld_obs[mask])[(ld0-15<ld_obs[mask]) & (ld_obs[mask]<ld0+15)])
        ##norm=np.percentile(flux,90)#1#40000

        # normalize by the flux at ld0
        f_obs_nu = f_obs_nu / f_obs0
        f_obserr_nu = f_obserr_nu / f_obs0
        #f_ref = norm_flux(path_obs, ld0)
    else:
        f_obs_nu = f_obs_lmd 
        f_obserr_nu = f_obserr_lmd


    path_telluric = '/home/yuikasagi/Develop/exojax/data/transmit_bound15_winter_all_cp.dat'
    telluric = pd.read_csv(path_telluric,header=None,sep='\s+')
    wav_telluric = telluric[0].values*1.e1 ##[AA]
    flux_telluric = telluric[4].values

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
                if len(dat_mask)<250:
                    ind_str, ind_end = dat_mask.index[0], dat_mask.index[-1]
                elif ord_list[k][j]>51:##h band
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
                ##mask telluric
                flux_telluric_interp = np.interp(ld_obs_j,wav_telluric,flux_telluric)
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
                    #print(ind,np.sum(mask_err))
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

    if norm:
        return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, ld0, wav_sort, flux_med #, f_ref
    else:
        return ld_obs_l, f_obs_nu_l, f_obserr_nu_l, ord_l, ld0#, f_ref
    
def spec_kasagi_norm(path_spec):
    dat = pd.read_csv(path_spec)

    wav_sort, flux_sort = [], []
    for order in dat['order'].unique():
        dat_ord = dat[dat['order']==order]
        wav_sort.extend(dat_ord['wav'][200:-300].values)
        flux_sort.extend(dat_ord['flux'][200:-300].values)
    wav_sort = np.array(wav_sort)
    flux_sort = np.array(flux_sort)

    flux_med = medfilt(flux_sort,kernel_size=999)

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

def read_telluric(path_spec,wavlim=[14000,17500]):
    #path_telluric = '/home/yuikasagi/Develop/exojax/data/transmit_bound15_winter_all_cp.dat'
    telluric = pd.read_csv(path_spec,header=None,sep='\s+')
    wav_telluric = telluric[0].values*1.e1 ##[AA]
    flux_telluric = telluric[4].values
    ind = (wavlim[0]<=wav_telluric) & (wav_telluric<=wavlim[1])
    wav_telluric = wav_telluric[ind][::-1]
    flux_telluric = flux_telluric[ind][::-1]
    return wav_telluric,flux_telluric