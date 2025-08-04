"""Cross-Correlation Function (CCF) of observed spectrum and best-fit model"""
# %%
import os
import numpy as np
from pathlib import Path
import setting
path_obs, path_data, path_telluric, path_save = setting.set_path()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

#------- setting ----------#
output_dir = Path("/home/yuikasagi/Develop/exojax/output/multimol/HR7672B/20210624/hmc_ulogg_nm/")

fit_cloud = True
order_use = [43, 44, 45, 57, 58, 59, 60]

ord_norm = {"y": 44, "h": 59}
#---------------------------#

# %%
#------------------------------------#
### load models, observed spectrum ###
#------------------------------------#
ord_list = [[x] for x in order_use]
band=[]
for k in range(len(ord_list)):
    if ord_list[k][0]<52:
        band.append(['y'])
    elif ord_list[k][0]>51:
        band.append(['h'])
band_unique = sorted(set(sum(band,[])))[::-1] ## sort y,h

from plotutils import load_data_all, load_data_model
order_connect = '-'.join([str(x) for x in order_use])
file_model = output_dir / f"models_order{order_connect}.npz"
data_model = np.load(file_model,allow_pickle=True)

## read data of data_model
_, _, ld_planet, _, f_speckle, _, model_wotel, transmitA, model_wo, mag_post = load_data_model(data_model, order_use, mle=False)
model_BD = [model_wotel[k]-f_speckle[k] for k in range(len(ld_planet))]

## load obs. spectrum (for Dp) and host star spectrum (for Ds)
import obs

path_spec,path_spec_A = {}, {}
path_spec_m1 = {}
for band_tmp in band_unique:
    path_spec[band_tmp] = os.path.join(path_obs,f'hr7672b/nwHR7672B_20210624_{band_tmp}_m2_photnoise.dat') 
    path_spec_A[band_tmp] = os.path.join(path_obs,f'hr7672a/nwHR7672A_20210606_{band_tmp}_m2_photnoise.dat')
    path_spec_m1[band_tmp] = os.path.join(path_obs,f'hr7672b/nwHR7672B_20210624_{band_tmp}_m1_photnoise.dat')

airmass_ratio = 1.024/1.148 # B/A = {20210625}/{20210607}

ld_obs,f_obs,f_obserr = [], [], []
ld_obs_A,f_obs_A,f_obserr_A = [], [], []
ld_obs_m1,f_obs_m1,f_obserr_m1 = [], [], []
for band_tmp in band_unique:
    ind_tmp = [x==band_tmp for x in sum(band,[])]
    # companion BD
    ld_obs_tmp, f_obs_tmp, f_obserr_tmp, ord_tmp, ld0_tmp = obs.spec(
        path_spec[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], 
        norm=False)

    # host star
    ld_obs_A_tmp, f_obs_A_tmp, f_obserr_A_tmp, _, _ = obs.spec(
        path_spec_A[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], 
        norm=False, lowermask=False, airmass_ratio=airmass_ratio)
    
    # speckle
    ld_obs_m1_tmp, f_obs_m1_tmp, f_obserr_m1_tmp, ord_m1_tmp, ld0_m1_tmp = obs.spec(
        path_spec_m1[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], 
        norm=False)

    ld_obs.extend(ld_obs_tmp)
    f_obs.extend(f_obs_tmp)
    f_obserr.extend(f_obserr_tmp)
    ld_obs_A.extend(ld_obs_A_tmp)
    f_obs_A.extend(f_obs_A_tmp)
    f_obserr_A.extend(f_obserr_A_tmp)
    ld_obs_m1.extend(ld_obs_m1_tmp)
    f_obs_m1.extend(f_obs_m1_tmp)
    f_obserr_m1.extend(f_obserr_m1_tmp)
# %%
fig,ax = plt.subplots()
for k in [-1]:#range(len(ld_planet)):
    ax.plot(ld_planet[k], model_BD[k], color="C0")
    #ax.plot(ld_planet, transmitA)
plt.show()

# %%
def interp_wavgrid(wav_base, wav, flux):
    flux_interp = []
    for k in range(len(wav_base)):
        interp_inv = np.interp(wav_base[k][::-1], wav[k][::-1], flux[k][::-1])
        flux_interp.append(interp_inv[::-1])
    return flux_interp

try:
    assert np.sum([np.sum(ld_obs[k] - ld_planet[k]) for k in range(len(ld_obs))]) == 0.0
except:
    print("Interpolating ld_obs to ld_planet grid...")
    model_BD = interp_wavgrid(ld_obs, ld_planet, model_BD)
    transmitA = interp_wavgrid(ld_obs, ld_planet, transmitA)

try:
    assert np.sum([np.sum(ld_obs[k] - ld_obs_A[k]) for k in range(len(ld_obs))]) == 0.0
except:
    print("Interpolating ld_obs_A to ld_obs grid...")
    f_obs_A = interp_wavgrid(ld_obs, ld_obs_A, f_obs_A)

try:
    assert np.sum([np.sum(ld_obs[k] - ld_obs_m1[k]) for k in range(len(ld_obs))]) == 0.0
except:
    print("Interpolating ld_obs_m1 to ld_obs grid...")
    f_obs_m1 = interp_wavgrid(ld_obs, ld_obs_m1, f_obs_m1)    

# %%
## cut order edges and flatten
ld_obs_cut, f_obs_cut, f_obserr_cut = [], [], []
f_obs_A_cut = []
model_BD_cut, transmitA_cut = [], []
f_obs_m1_cut = []
cut_pix_obs = 50

for k in range(len(order_use))[:]:
    ld_obs_cut.extend(ld_obs[k][cut_pix_obs:-cut_pix_obs])
    f_obs_cut.extend(f_obs[k][cut_pix_obs:-cut_pix_obs])
    f_obserr_cut.extend(f_obserr[k][cut_pix_obs:-cut_pix_obs])
    f_obs_A_cut.extend(f_obs_A[k][cut_pix_obs:-cut_pix_obs])
    model_BD_cut.extend(model_BD[k][cut_pix_obs:-cut_pix_obs])
    transmitA_cut.extend(transmitA[k][cut_pix_obs:-cut_pix_obs])
    f_obs_m1_cut.extend(f_obs_m1[k][cut_pix_obs:-cut_pix_obs])

ld_obs_cut = np.array(ld_obs_cut)
f_obs_cut = np.array(f_obs_cut)
f_obserr_cut = np.array(f_obserr_cut)
f_obs_A_cut = np.array(f_obs_A_cut)
model_BD_cut = np.array(model_BD_cut)
transmitA_cut = np.array(transmitA_cut)
f_obs_m1_cut = np.array(f_obs_m1_cut)

def sort_list(wav, flux, ferr):
    sort_ind = np.argsort(wav)
    return wav[sort_ind], flux[sort_ind], ferr[sort_ind]

_, model_BD_sort, _ = sort_list(ld_obs_cut, model_BD_cut, np.zeros_like(ld_obs_cut))
_, transmitA_sort, _ = sort_list(ld_obs_cut, transmitA_cut, np.zeros_like(ld_obs_cut))
ld_obs_sort, f_obs_sort, f_obserr_sort = sort_list(ld_obs_cut, f_obs_cut, f_obserr_cut)
_, f_obs_A_sort, _ = sort_list(ld_obs_cut, f_obs_A_cut, np.zeros_like(ld_obs_cut))
_, f_obs_m1_sort, _ = sort_list(ld_obs_cut, f_obs_m1_cut, np.zeros_like(ld_obs_cut))
# %%
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(ld_obs_sort, model_BD_sort)
plt.show()
# %%
#-----------------------------------------#
### cross-correlation function analysis ###
#-----------------------------------------#
import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import lstsq

def doppler_shift(wave, flux, rv):
    """
    Apply Doppler shift to flux at given wavelengths.
    rv: radial velocity in km/s
    """
    c = 299792.458  # speed of light in km/s
    shift_factor = np.sqrt((1 + rv / c) / (1 - rv / c))
    new_wave = wave * shift_factor
    interp_func = interp1d(new_wave, flux, bounds_error=False, fill_value=(flux[0], flux[-1]))
    return interp_func(wave)

# high-pass filter to subtract continuum
from scipy.ndimage import median_filter

def high_pass_filter(flux, size=200):
    continuum = median_filter(flux, size=size)
    #fig, ax = plt.subplots()
    #ax.plot(flux)
    #ax.plot(continuum)
    #plt.show()
    flux_hpf = flux - continuum
    return flux_hpf #- np.mean(flux_hpf)


def compute_ccf(observed, obserr, wave, tpl_planet, tpl_transmit, tpl_star, rv_grid, hpf_size=200):
    """
    Compute Ruffio-style CCF using least squares after high-pass filtering.
    """
    # Apply high-pass filtering to data and stellar template (independent of RV)
    observed_hpf = high_pass_filter(observed, size=hpf_size)
    tpl_star_hpf = high_pass_filter(tpl_star, size=hpf_size)

    if obserr is not None:
        weight = 1. / obserr**2
        W = np.diag(weight)
    print(np.mean(observed_hpf))

    ccf_values = []
    x_all = []
    for rv in rv_grid:
        # Doppler shift planet template and apply high-pass filter
        tpl_p_shifted = doppler_shift(wave, tpl_planet, rv)
        tpl_tp = tpl_p_shifted * tpl_transmit
        tpl_p_hpf = high_pass_filter(tpl_tp, size=hpf_size)

        # Stack model matrix A: shape (N_lambda, 2)
        A = np.vstack([tpl_p_hpf, tpl_star_hpf]).T 

        # Solve A x = y by least squares: x = [alpha_p, alpha_s]
        if obserr is None:
            x, _, _, _ = lstsq(A, observed_hpf, rcond=None)
        else:
            AtW = A.T * weight  # broadcasting
            x = np.linalg.solve(AtW @ A, AtW @ observed_hpf)

        # Store estimated alpha_p (planet scaling)
        ccf_values.append(x[0])
        x_all.append(x)

        if rv == -500:
            print(x)
            fig,ax = plt.subplots(figsize=(15, 5))
            #ax.plot(wave, observed, alpha=0.5)
            ax.plot(wave, observed_hpf, label='Observed (HPF)', alpha=0.5)
            ax.plot(wave, tpl_p_hpf, label='Planet Template (HPF)', alpha=0.5)
            ax.plot(wave, tpl_star_hpf, label='Star Template (HPF)', alpha=0.5)
            ax.legend()
            #ax.set(xlim=(15000, 15200))#(12850, 12950))#
            plt.show()

    return rv_grid, np.array(ccf_values), np.array(x_all)

# %%
rvmin, rvmax, rvstep = -1000., 1000., 1.
rv_grid = np.arange(rvmin, rvmax, rvstep)

ccf, ccf_m1 = np.zeros_like(rv_grid), np.zeros_like(rv_grid)
acf = np.zeros_like(rv_grid)
for band in band_unique:
    if band == 'y':
        mask = ld_obs_sort < 14000
    elif band == 'h':
        mask = ld_obs_sort > 14000
    #high_pass_filter(model_BD_sort[mask], size=1500)
    rv, ccf_tmp, _ = compute_ccf(f_obs_sort[mask], f_obserr_sort[mask], ld_obs_sort[mask], model_BD_sort[mask], transmitA_sort[mask], f_obs_A_sort[mask], rv_grid, hpf_size=1500)
    _, ccf_m1_tmp, _ = compute_ccf(f_obs_m1_sort[mask], None, ld_obs_sort[mask], model_BD_sort[mask], transmitA_sort[mask], f_obs_A_sort[mask], rv_grid, hpf_size=1500)
    _, acf_tmp, _ = compute_ccf(f_obs_sort[mask], f_obserr_sort[mask], ld_obs_sort[mask], f_obs_sort[mask], transmitA_sort[mask], f_obs_A_sort[mask], rv_grid, hpf_size=1500)
    ccf += ccf_tmp
    ccf_m1 += ccf_m1_tmp
    acf += acf_tmp

# %%
noise_region = (rv < -500) | (rv > 500)
sigma_ccf_noise = np.std(ccf[noise_region])
mean_ccf_noise = np.mean(ccf[noise_region])

maxind = 1000
print(((ccf-mean_ccf_noise)/sigma_ccf_noise)[maxind])

fig,ax=plt.subplots(figsize=(8, 5))
ax.plot(rv, (ccf-mean_ccf_noise)/sigma_ccf_noise)
ax.plot(rv, (ccf_m1-np.mean(ccf_m1))/sigma_ccf_noise, color="grey", alpha=0.6, zorder=1)
#ax.plot(rv, acf / sigma_ccf_noise)#(acf-mean_ccf_noise)/sigma_ccf_noise)
ax.axvline(0, color="k", ls="dashed", lw=1.)

ax.legend(
    [f"CCF of data & best-fit retrieval model", 
     f"CCF of speckle & best-fit retrieval model",
    ],
     bbox_to_anchor=(0., -0.2), loc='upper left'#, ncols=2
     )
ax.set(xlabel="Velocity shift [km/s]", ylabel="S/N of CCF")
ax.set(xlim=(rvmin, rvmax))
#plt.show()
#plt.savefig(output_dir / f"ccf_bestfit.png", bbox_inches='tight')

# %%
