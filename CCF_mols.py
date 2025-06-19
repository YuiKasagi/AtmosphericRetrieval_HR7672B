"""Cross-Correlation Function (CCF) to demonstrate the detection of FeH"""
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

band = "y"
fit_cloud = True
order_use = [43, 44, 45]#[58, 59, 60]#
order_all = [43, 44, 45, 57, 58, 59, 60]
path_spec = os.path.join(path_obs, f'hr7672b/nwHR7672B_20210624_{band}_m2_photnoise.dat') 

save = False
brv = 13.33 # barycentric velocity 

if band=="y":
    ord_norm = 44
    mol_use = [["FeH"]]*len(order_use)
    db_use = [["ExoMol"]]*len(order_use)    
elif band=="h":
    ord_norm = 59
    mol_use = [["H2O"]]*len(order_use)
    db_use = [["ExoMol"]]*len(order_use)

#---------------------------#

# %%
#------------------------------------#
### load models, observed spectrum ###
#------------------------------------#
from plotutils import load_data_all, load_data_model
order_connect = '-'.join([str(x) for x in order_all])
file_all = output_dir / f"all_order{order_connect}.npz"
file_model = output_dir / f"models_order{order_connect}.npz"

data_all = np.load(file_all,allow_pickle=True)
data_model = np.load(file_model,allow_pickle=True)

## read data of data_all
flux_median_mu, flux_hpdi_mu, mag_median_mu, mag_hpdi_mu = load_data_all(data_all, order_all)

## read data of data_model
name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_post, model_wotel, transmitA, model_wo, mag_post = load_data_model(data_model, order_all, mle=False)

## load flux uncertainties
import obs
order_use_tmp = [[x] for x in order_use]
ld_obs_tmp, f_obs_tmp, f_obserr_tmp, ord_tmp, ld0_tmp = obs.spec(path_spec, path_telluric, band, order_use_tmp, ord_norm=ord_norm, norm=False)


f_rm_fullmodel, f_rm_mol = [], []
for k in range(len(ord_list)):
    transmit_k = (flux_median_mu[k].astype(float)-f_speckle[k]) / np.array(model_wotel[k]-f_speckle[k])
    #f_obs_BD_k = (f_obs[k]-f_speckle[k]) / transmit_k  # only BD spectrum
    f_obs_BD_k = f_obs[k] # observed spectrum

    # F_rm_fullmodel = F_obs_BD - F_model_allmol
    f_model_allmol_k = model_wotel[k] - f_speckle[k]
    f_rm_fullmodel_k = f_obs_BD_k - f_model_allmol_k

    # F_rm_mol = F_obs_BD - F_model_w/o_mol
    for i, mol_tmp in enumerate(name_atommol_masked[k]):
        if mol_tmp in set(sum(mol_use, [])):
            print(mol_tmp)
            f_model_wo_mol_k = model_wo[i][k]-f_speckle[k]
            f_rm_mol_k = f_obs_BD_k - f_model_wo_mol_k
    f_rm_fullmodel.append(f_rm_fullmodel_k)
    f_rm_mol.append(f_rm_mol_k)


# %%
#------------------------------#
### create template spectrum ###
#------------------------------#
# order mask
assert sum(ord_list, []) == order_all
k_use = [sum(ord_list, []).index(x) for x in order_use]
mask_k = next(i for i, sublist in enumerate(ord_list) if ord_norm in sublist)
ld0 = np.nanmedian(ld_obs[mask_k])

## nu grid
import math
import jax.numpy as jnp
import jax
from exojax.utils.grids import wavenumber_grid
from exojax.spec.multimol import MultiMol
from exojax.spec import contdb
from exojax.spec.layeropacity import layer_optical_depth, layer_optical_depth_CIA
from exojax.spec.atmrt import ArtEmisPure
from exojax.spec import planck, response
from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.grids import velocity_grid
from utils import powerlaw_temperature_ptop

R = 1000000. # 10 x instrumental spectral resolution
nu_grid_list = []
wav_list = []
res_list = []
nusd = []
for k in k_use:
    wl_min = np.min(ld_obs[k])
    wl_max = np.max(ld_obs[k])
    nu_min = 1.0e8/(wl_max+5.0)
    nu_max = 1.0e8/(wl_min-5.0)
    Nx = math.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki
    Nx = math.ceil(Nx/2.) * 2 # make even
    nus_k, wav_k, res_k = wavenumber_grid(wl_min-5.0, wl_max+5.0, Nx, 
                                          unit="AA", xsmode="premodit", wavelength_order="ascending")
    nusd_k = jnp.array(1.0e8/ld_obs[k]) 
    nu_grid_list.append(nus_k)
    wav_list.append(wav_k)
    res_list.append(res_k)
    nusd.append(nusd_k)

# mol template
mul = MultiMol(molmulti=mol_use, dbmulti=db_use, database_root_path=path_data)

Tlow = 500.
Thigh = 4000.
Ttyp = 1000.
dit_grid_resolution = 1.
multimdb = mul.multimdb(nu_grid_list, crit=1.e-27, Ttyp=Ttyp)
multiopa = mul.multiopa_premodit(multimdb, nu_grid_list, auto_trange=[Tlow, Thigh], 
                                 dit_grid_resolution=dit_grid_resolution, allow_32bit=True)

# CIA
cdbH2H2=[]
cdbH2He=[]
for k in range(len(k_use)):
    cdbH2H2.append(contdb.CdbCIA(os.path.join(path_data,'H2-H2_2011.cia'),nu_grid_list[k]))
    cdbH2He.append(contdb.CdbCIA(os.path.join(path_data,'H2-He_2011.cia'),nu_grid_list[k]))

molmass_list, molmassH2, molmassHe = mul.molmass()
# %% 
NP = 300
# resolution
Rinst = 100000. #instrumental spectral resolution
beta_inst = resolution_to_gaussian_std(Rinst)  #equivalent to beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R)

# velocity grid
vsini_max = 100.0
vr_array = []
for k in range(len(k_use)):
    vr_array.append(velocity_grid(res_list[k], vsini_max))

# number of layers
NP = 300
# fixed values for telluric art
Tfix = 273.
Pfix = 0.6005

def frun(T0, alpha, logg, logvmr, RV, a, b, logPtop, taucloud, ign, obs_grid):
    """Same as frun in main_hmc.py, but without telluric
    """
    g = 10.**logg # cgs

    # volume mixing ratios
    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass_list)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass_list)) / mmw

    mu = []
    ld0_tmp = ld0
    ord_norm_tmp = ord_norm
    for k in range(len(k_use)):
        k_use_tmp = k_use[k]
        # Atmospheric setting by "art"
        art = ArtEmisPure(nu_grid = nu_grid_list[k], 
                          pressure_top = 1.e-3, 
                          pressure_btm = 1.e3, 
                          nlayer = NP,)
        art.change_temperature_range(Tlow, Thigh)
        if fit_cloud:
            Tarr = powerlaw_temperature_ptop(art.pressure, logPtop, T0, alpha)
        else:
            Tarr = art.powerlaw_temperature(T0, alpha)
        Parr = art.pressure
        dParr = art.dParr
        ONEARR=np.ones_like(Parr)

        # molecules
        dtaum=[]
        for i in range(len(mul.masked_molmulti[k])):
            if(mul.masked_molmulti[k][i] not in ign):
                xsm = multiopa[k][i].xsmatrix(Tarr, Parr)
                xsm=jnp.abs(xsm)
                dtaum.append(layer_optical_depth(dParr, xsm, mmr[mul.mols_num[k][i]]*ONEARR, 
                                                 molmass_list[mul.mols_num[k][i]], g))

        dtau = sum(dtaum)

        # CIA
        if(len(cdbH2H2[k].nucia) > 0):
            dtaucH2H2 = layer_optical_depth_CIA(nu_grid_list[k], Tarr, Parr, dParr, vmrH2, vmrH2, mmw, 
                                                g, cdbH2H2[k].nucia, cdbH2H2[k].tcia, cdbH2H2[k].logac)
            dtau = dtau + dtaucH2H2
        if(len(cdbH2He[k].nucia) > 0):
            dtaucH2He=layer_optical_depth_CIA(nu_grid_list[k], Tarr, Parr, dParr, vmrH2, vmrHe, mmw, 
                                              g, cdbH2He[k].nucia, cdbH2He[k].tcia, cdbH2He[k].logac)
            dtau = dtau + dtaucH2He

        # cloud
        if fit_cloud:
            tau0 = jax.nn.sigmoid(NP*5*(jnp.log10(Parr) - (logPtop)))*(taucloud)
            dtau_cloud = tau0[:,None]
            dtau += dtau_cloud

        # BD spectrum
        sourcef = planck.piBarr(Tarr,nu_grid_list[k])
        F0 = rtrun_emis_pureabs_fbased2st(dtau,sourcef)
        f_normalize = a + b*(nu_grid_list[k]-1.e8/ld0_tmp) # linear continuum
        F0 = F0/f_normalize
        Frot = F0 #convolve_rigid_rotation(F0, vr_array[k], vsini, u1, u2)

        # response
        if obs_grid:
            mu_k = response.ipgauss_sampling(nusd[k], nu_grid_list[k], Frot, beta_inst, RV, vr_array[k])
            if ord_norm_tmp == order_use[k]:    
                f_obs0 = jnp.nanmedian(mu_k[(ld0_tmp-15<ld_obs[k_use_tmp]) & (ld_obs[k_use_tmp]<ld0_tmp+15)])
        else:
            mu_k = response.ipgauss_sampling(nu_grid_list[k], nu_grid_list[k], Frot, beta_inst, RV, vr_array[k])
            #mu_k = F0
            mu_itp = jnp.interp(ld_obs[k_use_tmp][::-1], wav_list[k], mu_k[::-1]) # wav ascending order 
            mu_itp = mu_itp[::-1]
            if ord_norm_tmp == order_use[k]:
                f_obs0 = jnp.nanmedian(mu_itp[(ld0_tmp-15<ld_obs[k_use_tmp]) & (ld_obs[k_use_tmp]<ld0_tmp+15)])

        mu_k = mu_k
        mu.append(mu_k)

    return mu, jnp.abs(f_obs0)

# %%
#-------------------------------#
### load retrieved parameters ###
#-------------------------------#

import pickle
file_samples = output_dir / f"samples_order{order_connect}_1000.pickle"
with open(file_samples, mode='rb') as f:
    samples = pickle.load(f)

T0 = np.median(samples["T0"])
alpha = np.median(samples["alpha"])
logg = np.median(samples["logg"])
logvmr = []
for mol_use_tmp in mol_use[0]:
    logvmr.append(np.median(samples[f"log{mol_use_tmp}"]))
RV = np.median(samples["RV"])
vsini = 0.1 #np.median(samples["vsini"])
a = np.median(samples["a_y"]) #j band
b = np.median(samples["b_y"]) * 1e-2#j band
if fit_cloud:
    logPtop = np.median(samples["logPtop"])
else:
    logPtop = None

mu, f_obs0 = frun(T0=T0, alpha=alpha, logg=logg, logvmr=logvmr, RV=RV, a=a, b=b, logPtop=logPtop, taucloud=500, ign="ign", obs_grid=False)
f_template = []
for k in range(len(mu)):
    f_template.append(mu[k] / f_obs0)

# %%
#------------------#
### plot spectra ###
#------------------#

fig, ax = plt.subplots(figsize=(15,5))
ld_obs_cut, f_rm_fullmodel_cut, f_rm_mol_cut = [], [], []
f_obserr_cut = []
wav_template_cut, f_template_cut = [], []
for k_tmp,k in enumerate(k_use):
    ax.plot(ld_obs[k], f_rm_fullmodel[k], label=f"Full model {ord_list[k]}", alpha=0.5)
    ax.plot(ld_obs[k], f_rm_mol[k], label=f"Without {mol_use[0][0]} {ord_list[k]}", alpha=0.5)
    ax.plot(wav_list[k_tmp][::-1], f_template[k_tmp], label=f"Template {ord_list[k]}")

    cut_pix_obs = 50
    cut_pix_template = 200
    ld_obs_cut.extend(ld_obs[k][cut_pix_obs:-cut_pix_obs])
    f_rm_fullmodel_cut.extend(f_rm_fullmodel[k][cut_pix_obs:-cut_pix_obs])
    f_rm_mol_cut.extend(f_rm_mol[k][cut_pix_obs:-cut_pix_obs])
    if np.all(ld_obs_tmp[k_tmp] == ld_obs[k]):
        f_obserr_cut.extend(f_obserr_tmp[k_tmp][cut_pix_obs:-cut_pix_obs])
    else:
        print("WARNING: NO ERROR")
        continue
    wav_template_cut.extend(wav_list[k_tmp][::-1][cut_pix_template:-cut_pix_template])
    f_template_cut.extend(f_template[k_tmp][cut_pix_template:-cut_pix_template])

ld_obs_cut = np.array(ld_obs_cut)
f_rm_fullmodel_cut = np.array(f_rm_fullmodel_cut)
f_rm_mol_cut = np.array(f_rm_mol_cut)
f_obserr_cut = np.array(f_obserr_cut)
wav_template_cut = np.array(wav_template_cut)
f_template_cut = np.array(f_template_cut)

plt.show()

# %%

sort_obs = np.argsort(ld_obs_cut)
sort_template = np.argsort(wav_template_cut)

# data
dw = ld_obs_cut[sort_obs]
df_rm_fullmodel = f_rm_fullmodel_cut[sort_obs] 
df_rm_mol = f_rm_mol_cut[sort_obs] 
df_err = f_obserr_cut[sort_obs]

# template
tw = wav_template_cut[sort_template]
tf = f_template_cut[sort_template]
tf /= np.median(tf)

import pandas as pd
df_obs = pd.DataFrame(np.array([dw, df_rm_fullmodel, df_rm_mol, df_err]).T, columns=["dw", "df_rm_fullmodel", "df_rm_mol", "df_err"])

"""
# binning
wobs_min, wobs_max = df_obs["dw"].min(), df_obs["dw"].max()
delta_wav = (wobs_max - wobs_min) / 4200 
df_obs['bin'] = ((df_obs['dw'] - wobs_min) // delta_wav).astype(int)

def combine_error(g):
    return np.sqrt(np.sum(g**2)) / len(g)

df_obs_binned = df_obs.groupby('bin').agg({
    'dw': 'mean',
    'df_rm_fullmodel': 'mean',
    'df_rm_mol': 'mean',
    'df_err': combine_error
}).reset_index(drop=True)
"""
# no binning
df_obs_binned = df_obs 
dw = df_obs_binned["dw"].values
df_rm_fullmodel = df_obs_binned["df_rm_fullmodel"].values
df_rm_mol = df_obs_binned["df_rm_mol"].values
df_err = df_obs_binned["df_err"].values

# %%
#-----------------------------------------#
### cross-correlation function analysis ###
#-----------------------------------------#
from astropy.constants import c
from scipy.interpolate import interp1d

c_kms = c.to("km/s").value
rv_grid = np.arange(-100, 101, 1)

def ccf_scaling(wavelength_template, 
                flux_template,
                wavelength_data,
                flux_data,
                uncert_data,
                rv_grid):
    ccf_flux = []
    # template
    interp_template = interp1d(wavelength_template, flux_template,
                            kind='linear', bounds_error=False, fill_value=0.0)

    # CCF for rv
    for rv in rv_grid:
        # RV shift
        shifted_wavelength = wavelength_data / (1 + rv / c_kms)
        shifted_template_flux = interp_template(shifted_wavelength)
        
        # weighted linear fit
        w = 1.0 / (uncert_data**2)
        numerator = np.sum(w * shifted_template_flux * flux_data)
        denominator = np.sum(w * shifted_template_flux**2)
        
        # scaling factor
        if denominator > 0:
            a_v = numerator / denominator
        else:
            a_v = 0.0
        ccf_flux.append(a_v)
    return ccf_flux

ccf_flux_rm_fullmodel = ccf_scaling(tw, tf, dw, df_rm_fullmodel, df_err, rv_grid)
ccf_flux_rm_mol = ccf_scaling(tw, tf, dw, df_rm_mol, df_err, rv_grid)

# %%
#----------------------#
### plot CCF results ###
#----------------------#

fig, axs = plt.subplots(2,1,figsize=(15,5), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.)
axs[1].plot(dw, df_rm_fullmodel, label=f"data - model w/ {mol_use[0][0]}", lw=1.5, alpha=0.7)
axs[1].plot(dw, df_rm_mol, label=f"data - model w/o {mol_use[0][0]}", lw=1.5, alpha=0.7)
#axs[1].plot(dw, df_err, '.', color="grey")
axs[0].plot(tw, tf, label=f"{mol_use[0][0]} template", color="k", lw=1.5, alpha=0.7)
axs[0].legend(loc="lower right")
if band=="y":
    axs[1].legend(loc="lower right", ncols=2)
elif band=="h":
    axs[1].legend(loc="upper right", ncols=2)
axs[0].set(xlim=(np.min(tw),np.max(tw)))
#axs[0].set(ylabel="normalized flux")
axs[1].set(xlabel="Wavelength [$\AA$]", ylabel="Normalized flux")

ord_str = [str(ord) for ord in order_use]
num = '-'.join(ord_str)
if save:
    plt.savefig(output_dir / f"template_spec_order{num}.png", bbox_inches='tight')#,dpi=300)
else:
    plt.show()
# %%
fig,ax1=plt.subplots(figsize=(7,5))
ax2 = ax1.twinx()
ax1.plot(rv_grid, ccf_flux_rm_fullmodel, color="C0", alpha=0.8)
ax1.plot(-200, 1, color="C1", alpha=0.8)
ax2.plot(rv_grid, ccf_flux_rm_mol, color="C1", alpha=0.8)

ax1.axvline(0,ls='--',color="k", lw=1.5)
ax1.axvline(brv,ls='--',color="grey", lw=1.5)
ax1.legend(
    [f"CCF of (data - model w/ {mol_use[0][0]}) & {mol_use[0][0]} template (Left y-axis)", 
     f"CCF of (data - model w/o {mol_use[0][0]}) & {mol_use[0][0]} template (Right y-axis)"],
     bbox_to_anchor=(-0.25, -0.2), loc='upper left'#, ncols=2
     )
ax1.set(xlabel="Velocity shift [km/s]", ylabel="Normalized flux")
ax1.set(xlim=(-100,100),ylim=(np.min(ccf_flux_rm_fullmodel)-0.001, np.min(ccf_flux_rm_fullmodel)+0.005))
ax2.set(ylim=(np.min(ccf_flux_rm_mol)-0.001, np.min(ccf_flux_rm_mol)+0.005))

if save:
    plt.savefig(output_dir / f"ccf_order{num}.png", bbox_inches='tight')#,dpi=300)
else:
    plt.show()

# %%
