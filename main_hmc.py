"""
HMC sample for multiple molecules, originally written by @ykawashima, 
modified by @HajimeKawahara using exojax.spec.multimol,
modified by @YuiKasagi for Kasagi et al. (2025)
"""
import time
ts = time.time()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
sys.path.insert(0,'/home/yuikasagi/exojax/src')
sys.path.insert(0,'/home/yuikasagi/radis')

#multimol
from exojax.spec.multimol import MultiMol
import numpy as np
import jax.numpy as jnp
import jax
import os
from jax import config
config.update("jax_enable_x64", False)

#-----------SETTING-----------#
def init(lang):
    return [int(x) for x in lang.split("-")]

import argparse
parser = argparse.ArgumentParser(description="hmc")
parser.add_argument('-o', '--order', nargs="*", default=[[43, 44, 45, 57, 58, 59, 60]], type=init, help="echelle order")
parser.add_argument('-t', '--target', nargs=1, default=['HR7672B'], type=str, help="target name")
parser.add_argument('-d', '--date', nargs=1, default=[20210624], type=int, help="observation date")
parser.add_argument('--mmf', nargs=1, default=['m2'], type=str, help="used fiber (m1 or m2)")
parser.add_argument('--fit_cloud', action='store_true', help="fit cloud")
parser.add_argument('--fit_speckle', action='store_true', help="fit speckle")
parser.add_argument('--run', action='store_true', help="run mcmc")

args = parser.parse_args()
order=args.order[0]
target=args.target[0]
date=args.date[0]
mmf = args.mmf[0]
fit_cloud = args.fit_cloud
fit_speckle = args.fit_speckle
run = args.run

ord_list = [[x] for x in order]

print('ord_list = ',ord_list)
band=[]
for k in range(len(ord_list)):
    if ord_list[k][0]<52:
        band.append(['y'])
    elif ord_list[k][0]>51:
        band.append(['h'])
band_unique = sorted(set(sum(band,[])))[::-1] ## sort y,h

## Paths
import setting
path_obs, path_data, path_telluric, path_save = setting.set_path()
save_dir = path_save+"/multimol/"+target+"/"+str(date)+"/hmc_ulogg_nm_tmp/"  ##CHECK!!

## High-resolution Spectra
mol0, db0 = [],[]
mol_tel0, db_tel0 = [],[]
ord_norm = {}
for k in range(len(ord_list)):
    if band[k][0] == 'y':
        # BD
        mol0.append(["H2O","FeH"])
        db0.append(["ExoMol","ExoMol"])
        # telluric
        mol_tel0.append(["H2O","CH4","CO2","O2"])
        db_tel0.append(["ExoMol","HITEMP","ExoMol","HITRAN12"])
        ord_norm["y"] = 44 
    elif band[k][0] == 'h':
        # BD
        mol0.append(["H2O"])
        db0.append(["ExoMol"])
        # telluric
        mol_tel0.append(["H2O","CH4","CO2"])
        db_tel0.append(["ExoMol","HITEMP","ExoMol"])
        ord_norm["h"] = 59 
mul = MultiMol(molmulti=mol0, dbmulti=db0, database_root_path=path_data)
mul_tel = MultiMol(molmulti=mol_tel0, dbmulti=db_tel0, database_root_path=path_data)

## Photometry
num_ord_list_p = 1 # -> y only
mol_p, db_p = [], []
for band_tmp in band_unique:
    if band_tmp=='y':
        mol_p.append(["H2O","FeH"])
        db_p.append(["ExoMol","ExoMol"])
mul_p = MultiMol(molmulti=mol_p, dbmulti=db_p, database_root_path=path_data)


## Initial Parameters
path_spec,path_spec_A = {}, {}
if target=="HR7672B":
    # T0, alpha, RV, vsini, vtel
    boost0 = np.array([2100., 0.1, -20., 45., 1.])
    initpar0 = np.array([2100., 0.1, -20., 45., 0.])
    for band_tmp in band_unique:
        path_spec[band_tmp] = os.path.join(path_obs,f'hr7672b/nw{target}_{date}_{band_tmp}_{mmf}_photnoise.dat') 
        path_spec_A[band_tmp] = os.path.join(path_obs,f'hr7672a/nwHR7672A_20210606_{band_tmp}_{mmf}_photnoise.dat')
        # a, b
        boost0 = np.append(boost0,[1.,1.])
        initpar0 = np.append(initpar0,[1.,0.])
    # logg, Mp
    boost0 = np.append(boost0,[1.,1.]) 
    initpar0 = np.append(initpar0,[5.38, 72.7])

if fit_cloud:
    # logPtop
    boost0 = np.append(boost0,[1.])
    initpar0 = np.append(initpar0,[0.5])
    fix_taucloud = 500.

if fit_speckle:
    #for H
    if 'h' in band_unique:
        # logscale_star_h
        boost0 = np.append(boost0,[1.])#,1.])
        initpar0 = np.append(initpar0,[jnp.log10(0.5)])#, 0.])

    #for YJ
    """#if not fix
    if 'y' in band_unique:
        boost0 = np.append(boost0,[1.])#,1.])
        initpar0 = np.append(initpar0,[jnp.log10(0.5)])#, 0.])
    """
    fix_logscale_star = -0.20 # log_scale_star_y
    fix_relRV = 0. # relRV

fix_logg = 5.38
fix_Mp = 72.7 
#-----------------------------#

import obs
## Companion and Host Star
ld_obs,f_obs,f_obserr = [], [], []
ld_obs_A,f_obs_A,f_obserr_A = [], [], []
ld0 = {}
for band_tmp in band_unique:
    ind_tmp = [x==band_tmp for x in sum(band,[])]
    # companion BD
    ld_obs_tmp, f_obs_tmp, f_obserr_tmp, ord_tmp, ld0_tmp = obs.spec(
        path_spec[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], norm=False)

    # host star
    ld_obs_A_tmp, f_obs_A_tmp, f_obserr_A_tmp, _, _ = obs.spec(
        path_spec_A[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], norm=False, lowermask=False)

    ld_obs.extend(ld_obs_tmp)
    f_obs.extend(f_obs_tmp)
    f_obserr.extend(f_obserr_tmp)
    ld_obs_A.extend(ld_obs_A_tmp)
    f_obs_A.extend(f_obs_A_tmp)
    f_obserr_A.extend(f_obserr_A_tmp)
    ld0[band_tmp] = ld0_tmp

#if np.sum(np.array(ld_obs) - np.array(ld_obs_A))!=0:
#    print('Warning: wavelength grid has been different (A/B).')
f_obs_A_interp, f_obserr_A_interp = [], []
for k in range(len(ord_list)):
    # align the wavelength grid
    f_obs_A_interp_k = np.interp(ld_obs[k][::-1],ld_obs_A[k][::-1],f_obs_A[k][::-1])
    f_obserr_A_interp_k = np.interp(ld_obs[k][::-1],ld_obs_A[k][::-1],f_obserr_A[k][::-1])
    f_obs_A_interp.append(f_obs_A_interp_k[::-1])
    f_obserr_A_interp.append(f_obserr_A_interp_k[::-1])    
f_obs_A = f_obs_A_interp
f_obserr_A = f_obserr_A_interp

from exojax.utils.grids import wavenumber_grid
import math
R = 1000000. # 10 x instrumental spectral resolution
nu_grid_list = []
wav_list = []
res_list = []
nusd = []
for k in range(len(ord_list)):
    wl_min = np.min(ld_obs[k])
    wl_max = np.max(ld_obs[k])
    nu_min = 1.0e8/(wl_max+5.0)
    nu_max = 1.0e8/(wl_min-5.0)
    Nx = math.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki
    Nx = math.ceil(Nx/2.) * 2 # make even
    nus_k, wav_k, res_k = wavenumber_grid(wl_min-5.0, wl_max+5.0, Nx,unit="AA", xsmode="premodit", wavelength_order="ascending")
    nusd_k = jnp.array(1.0e8/ld_obs[k]) 
    nu_grid_list.append(nus_k)
    wav_list.append(wav_k)
    res_list.append(res_k)
    nusd.append(nusd_k)


## Photometry
## Boccaletti et al. 2003
J_mag_obs = 14.39 #appears to be corrupted by the huge speckle background (i.e. flux is overestimated)
J_mag_obserr = 0.20
H_mag_obs = 14.04
H_mag_obserr = 0.14

dict_mag_obs = {'y':J_mag_obs,'h':H_mag_obs}
dict_magerr_obs = {'y':J_mag_obserr,'h':H_mag_obserr}

import math
from scipy import interpolate
# photometry
# PHARO (Palomar) <-- not available?
# WFCAM (UKIDSS; UKIRT Deep Sky Survey) is referred, so we use WFCAM filter.
# c.f. https://sites.astro.caltech.edu/palomar/observer/200inchResources/sensitivities.html
def read_photometry_file(band):
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
    return wl_min,wl_max,wl_ref,tr_ref,R_p,Rinst_p

nus_p = []
wav_p = []
res_p = []
nusd_p = []
wavd_p = []
tr = {}
Rinst_p = {}
mag_obs, magerr_obs = [], []
for k in range(num_ord_list_p):
    wl_min, wl_max, wl_ref, tr_ref, R_p, Rinst_p_k = read_photometry_file(band_unique[k])
    nu_min = 1.0e8/(wl_max + 5.0)
    nu_max = 1.0e8/(wl_min - 5.0)
    Nx = math.ceil(R_p * np.log(nu_max/nu_min)) + 1 # ueki
    Nx = math.ceil(Nx/2.) * 2 # make even
    nus_k, wav_k, res_k = wavenumber_grid(wl_min-5., wl_max+5., Nx,unit="AA", xsmode="premodit", wavelength_order="ascending")
    nus_p.append(nus_k)
    wav_p.append(wav_k[::-1])
    res_p.append(res_k)

    mask_p = (1.0e8/nus_k >= wl_min) * (1.0e8/nus_k <= wl_max)
    nusd_p.append(nus_k[mask_p])
    wavd_p.append(1.0e8/nus_k[mask_p])

    f = interpolate.interp1d(wl_ref, tr_ref)
    tr_k = f(1.0e8/nus_k[mask_p])

    tr[band_unique[k]] = tr_k
    Rinst_p[band_unique[k]] = Rinst_p_k
    mag_obs.append([dict_mag_obs[band_unique[k]]])
    magerr_obs.append([dict_magerr_obs[band_unique[k]]])

#-----------OPACITY-----------#
## High-resolution Spectra
dit_grid_resolution = 1.

def multiopa_direct(self,
                    multimdb,
                    nu_grid_list):
    """multiple opa for OpaDirect
    """
    from exojax.spec.opacalc import OpaDirect
    multiopa = []
    for k in range(len(multimdb)):
        opa_k = []
        for i in range(len(multimdb[k])):
            multimdb[k][i].generate_jnp_arrays()
            opa_i = OpaDirect(mdb=multimdb[k][i],
                                nu_grid=nu_grid_list[k],
                                wavelength_order="ascending")
            opa_k.append(opa_i)
        multiopa.append(opa_k)

    return multiopa

# molecules
print('set opa for molecules')
Tlow = 500.
Thigh = 4000.
Ttyp = 1000.
multimdb = mul.multimdb(nu_grid_list, crit=1.e-27, Ttyp=Ttyp)
multiopa = mul.multiopa_premodit(multimdb, nu_grid_list, auto_trange=[Tlow,Thigh], dit_grid_resolution=dit_grid_resolution)

name_atommol = mul.mols_unique
name_atommol_masked = mul.masked_molmulti

# telluric
print('set opa for telluric')
Tlow_tel = 273.0
Thigh_tel = 1000.0
Ttyp_tel = 300.
multimdb_tel = mul_tel.multimdb(nu_grid_list, crit=1.e-27, Ttyp=Ttyp_tel)
multiopa_tel = multiopa_direct(mul_tel, multimdb_tel, nu_grid_list)

name_moltel = mul_tel.mols_unique
name_moltel_masked = mul_tel.masked_molmulti
    
# CIA
from exojax.spec import contdb
cdbH2H2=[]
cdbH2He=[]
for k in range(len(ord_list)):
    cdbH2H2.append(contdb.CdbCIA(os.path.join(path_data,'H2-H2_2011.cia'),nu_grid_list[k]))
    cdbH2He.append(contdb.CdbCIA(os.path.join(path_data,'H2-He_2011.cia'),nu_grid_list[k]))

molmass_list, molmassH2, molmassHe = mul.molmass()

## Photometry
# molecules
multimdb_p = mul_p.multimdb(nus_p, crit=1.e-27, Ttyp=Ttyp)
multiopa_p = mul_p.multiopa_premodit(multimdb_p, nus_p, auto_trange=[Tlow,Thigh], dit_grid_resolution=dit_grid_resolution)

name_atommol_p = mul_p.mols_unique
name_atommol_masked_p = mul_p.masked_molmulti
molmass_list_p, molmassH2, molmassHe = mul_p.molmass()

# CIA
cdbH2H2_p=[]
cdbH2He_p=[]
for k in range(num_ord_list_p):
    cdbH2H2_p.append(contdb.CdbCIA(os.path.join(path_data,'H2-H2_2011.cia'),nus_p[k]))
    cdbH2He_p.append(contdb.CdbCIA(os.path.join(path_data,'H2-He_2011.cia'),nus_p[k]))

#-----------OPTIMIZATUION-----------#

from astropy import constants as const
Mjup = const.M_jup.value
Rjup = const.R_jup.value
G_const = const.G.value
pc = const.pc.value

from exojax.spec.layeropacity import layer_optical_depth, layer_optical_depth_CIA
from exojax.spec.atmrt import ArtEmisPure
from exojax.spec import planck
from exojax.spec.rtransfer import rtrun_emis_pureabs_fbased2st
from exojax.spec import response

from exojax.utils.instfunc import resolution_to_gaussian_std
Rinst=100000. #instrumental spectral resolution
beta_inst=resolution_to_gaussian_std(Rinst)  #equivalent to beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R)
beta_inst_p={}
for band_tmp in band_unique:
    try:
        beta_inst_p[band_tmp]=resolution_to_gaussian_std(Rinst_p[band_tmp])
    except:
        beta_inst_p[band_tmp]=None

# response settings
from exojax.utils.grids import velocity_grid
from exojax.spec.spin_rotation import convolve_rigid_rotation
vsini_max = 100.0
vr_array = []
for k in range(len(ord_list)):
    vr_array.append(velocity_grid(res_list[k], vsini_max))
vr_array_p = []
for k in range(num_ord_list_p):
    vr_array_p.append(velocity_grid(res_p[k], vsini_max))

distance=17.71 #pc, Bailer-Jones+(2018)

NP = 300.
Tfix = 273.
Pfix = 0.6005

def nu_shift(nu, dv):
    c = 2.99792458e5 #[km/s]
    dnu = dv*nu/c
    return dnu

def powerlaw_temperature_ptop(pressure, logPtop, T0, alpha):
    Ptop = 10**(logPtop)
    return T0*(pressure/Ptop)**alpha

def frun(T0, alpha, logg, logvmr, u1, u2, RV, vsini, a, b, logbeta, vtel, logPtop, taucloud, ign, obs_grid, band_use):
    g = 10.**logg # cgs

    # volume mixing ratios
    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass_list)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass_list)) / mmw

    mu = []
    transmit = []
    k_use = [i for i,x in enumerate(sum(band,[])) if x==band_use]
    ld0_tmp = ld0[band_use]
    ord_norm_tmp = ord_norm[band_use]
    for k in k_use:
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

        dtaum=[]
        for i in range(len(mul.masked_molmulti[k])):
            if(mul.masked_molmulti[k][i] not in ign):
                xsm = multiopa[k][i].xsmatrix(Tarr, Parr)
                xsm=jnp.abs(xsm)
                dtaum.append(layer_optical_depth(dParr, xsm, mmr[mul.mols_num[k][i]]*ONEARR, molmass_list[mul.mols_num[k][i]], g))

        dtau = sum(dtaum)

        # CIA
        if(len(cdbH2H2[k].nucia) > 0):
            dtaucH2H2 = layer_optical_depth_CIA(nu_grid_list[k],Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2[k].nucia,cdbH2H2[k].tcia,cdbH2H2[k].logac)
            dtau = dtau + dtaucH2H2
        if(len(cdbH2He[k].nucia) > 0):
            dtaucH2He=layer_optical_depth_CIA(nu_grid_list[k], Tarr, Parr, dParr, vmrH2, vmrHe, mmw, g, cdbH2He[k].nucia, cdbH2He[k].tcia, cdbH2He[k].logac)
            dtau = dtau + dtaucH2He

        if fit_cloud:
            tau0 = jax.nn.sigmoid(NP*5*(jnp.log10(Parr) - (logPtop)))*(taucloud)
            dtau_cloud = tau0[:,None]
            dtau += dtau_cloud

        sourcef = planck.piBarr(Tarr,nu_grid_list[k])
        F0 = rtrun_emis_pureabs_fbased2st(dtau,sourcef)
        f_normalize = a + b*(nu_grid_list[k]-1.e8/ld0_tmp) 
        F0 = F0/f_normalize
        Frot = convolve_rigid_rotation(F0, vr_array[k], vsini, u1, u2)


        beta = jnp.power(10., jnp.array(logbeta))
        dtaut=[]
        for i in range(len(mul_tel.masked_molmulti[k])):
            xst = multiopa_tel[k][i].xsvector(Tfix, Pfix)
            xst=jnp.abs(xst)
            xst=xst*beta[i]
            dtaut.append(xst)
        dtaut = sum(dtaut)
        transmit_k = jnp.exp(-dtaut)

        if obs_grid:
            mu_k = response.ipgauss_sampling(nusd[k], nu_grid_list[k], Frot, beta_inst, RV, vr_array[k])
            transmit_k = response.ipgauss_sampling(nusd[k], nu_grid_list[k], transmit_k, beta_inst, vtel, vr_array[k])
            if 'telluric' not in ign:
                mu_k = mu_k * transmit_k
            if ord_norm_tmp in ord_list[k]:    
                f_obs0 = jnp.nanmedian(mu_k[(ld0_tmp-15<ld_obs[k]) & (ld_obs[k]<ld0_tmp+15)])
        else:
            mu_k = response.ipgauss_sampling(nu_grid_list[k], nu_grid_list[k], Frot, beta_inst, RV, vr_array[k])
            transmit_k = response.ipgauss_sampling(nu_grid_list[k], nu_grid_list[k], transmit_k, beta_inst, vtel, vr_array[k])
            if 'telluric' not in ign:
                mu_k = mu_k * transmit_k
            mu_itp = jnp.interp(ld_obs[k], nu_grid_list[k], mu_k)
            if ord_norm_tmp in ord_list[k]:
                f_obs0 = jnp.nanmedian(mu_itp[(ld0_tmp-15<ld_obs[k]) & (ld_obs[k]<ld0_tmp+15)])

        mu_k = mu_k
        mu.append(mu_k)
        transmit.append(transmit_k)

    return mu, jnp.abs(f_obs0), transmit

def scale_speckle(scale_star, relRV, obs_grid=True):
    f_speckle = []
    for k in range(len(ord_list)):
        dnu_star = nu_shift(nusd[k], relRV)
        if obs_grid:
            f_obs_A_shift = jnp.interp(nusd[k], nusd[k]+dnu_star, f_obs_A[k])
        else:
            f_obs_A_shift = jnp.interp(nu_grid_list[k], nusd[k]+dnu_star, f_obs_A[k])
        f_speckle.append(scale_star*f_obs_A_shift)
    return f_speckle

f0={}
f0['y'] = 2.98e-9 # [W/m^2/um] # for J band, https://www.gemini.edu/observing/resources/magnitudes-and-fluxes
f0['h'] = 1.15e-9 # [W/m^2/um] # kurohon, checked the referenced book in google book
def calc_photo(mu,band):
    mu = jnp.concatenate(mu)
    #mu = mu * f_ref # [erg/s/cm^2/cm^{-1}]
    # [erg/s/cm^2/cm^{-1}] => [erg/s/cm^2/cm]
    mu = mu / ((wavd_p[int(int(band=='h')*(num_ord_list_p-1))])*1.0e-8)**2.0e0
    # [erg/s/cm^2/cm] => [W/m^2/um]
    mu = mu * 1.0e-7 * 1.0e4 * 1.0e-4

    fdl = jnp.trapezoid(mu*tr[band], wavd_p[int(int(band=='h')*(num_ord_list_p-1))])
    dl = jnp.trapezoid(tr[band], wavd_p[int(int(band=='h')*(num_ord_list_p-1))])
    f = fdl / dl

    f0_use = f0[band]
 
    mag = -2.5 * jnp.log10(f / f0_use)

    return mag


def frun_p(T0, alpha, logg, Mp, logvmr, u1, u2, RV, vsini, logPtop, taucloud, band_use):
    g = 10.**logg # cgs
    g_mks = g * 1e-2
    Rp = jnp.sqrt(G_const * Mp * Mjup / g_mks) / Rjup

    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass_list_p)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass_list_p)) / mmw

    mu = []
    k_use = [int(int(band_use=='h')*(num_ord_list_p-1))] #set 0 for 'y' and 1 for 'h'
    for k in k_use:
        # Atmospheric setting by "art"
        art = ArtEmisPure(nu_grid=nus_p[k], 
                          pressure_top=1.e-3, 
                          pressure_btm=1.e3, 
                          nlayer=NP,)
        art.change_temperature_range(Tlow, Thigh)
        if fit_cloud:
            Tarr = powerlaw_temperature_ptop(art.pressure, logPtop, T0, alpha)
        else:
            Tarr = art.powerlaw_temperature(T0, alpha)
        Parr = art.pressure
        dParr = art.dParr
        ONEARR=np.ones_like(Parr)
        dtaum = []
        for i in range(len(mul_p.masked_molmulti[k])):
            xsm = multiopa_p[k][i].xsmatrix(Tarr, Parr)
            xsm = jnp.abs(xsm)
            dtaum.append(layer_optical_depth(dParr, xsm, mmr[mul_p.mols_num[k][i]]*ONEARR, molmass_list_p[mul_p.mols_num[k][i]], g))

        dtau = sum(dtaum)

        #CIA
        if(len(cdbH2H2_p[k].nucia) > 0):
            dtaucH2H2 = layer_optical_depth_CIA(nus_p[k], Tarr, Parr, dParr, vmrH2, vmrH2, mmw, g, cdbH2H2_p[k].nucia, cdbH2H2_p[k].tcia, cdbH2H2_p[k].logac)
            dtau = dtau + dtaucH2H2
        if(len(cdbH2He_p[k].nucia) > 0):
            dtaucH2He = layer_optical_depth_CIA(nus_p[k], Tarr, Parr, dParr, vmrH2, vmrHe, mmw, g, cdbH2He_p[k].nucia, cdbH2He_p[k].tcia, cdbH2He_p[k].logac)
            dtau = dtau + dtaucH2He

        if fit_cloud:
            tau0 = jax.nn.sigmoid(NP*5*(jnp.log10(Parr) - logPtop))*taucloud
            dtau_cloud = tau0[:,None]
            dtau += dtau_cloud

        sourcef = planck.piBarr(Tarr, nus_p[k])
        F0 = rtrun_emis_pureabs_fbased2st(dtau, sourcef)

        Frot = convolve_rigid_rotation(F0, vr_array_p[k], vsini, u1, u2)
        mu_k = response.ipgauss_sampling(nusd_p[k], nus_p[k], Frot, beta_inst_p[band_use], RV, vr_array_p[k])

        #distance correction from Kirkpatrick et al. 2012
        mu_k = mu_k * ((Rp * Rjup) /(distance * pc))**2

        mu.append(mu_k)

    mag = calc_photo(mu,band_use)
    return mu,mag


def model_opt(params, boost, ign="ign", obs_grid=True, norm=None):
    T0 = params[0]*boost[0]
    alpha = params[1]*boost[1]
    RV = params[2]*boost[2]
    vsini = params[3]*boost[3]
    vtel = params[4]*boost[4]
    a,b=[],[]
    for i in range(len(band_unique)):
        a.append(params[5+2*i]*boost[5+2*i])
        b.append(params[6+2*i]*boost[6+2*i]*1e-2)
    logg = params[6+2*i+1]*boost[6+2*i+1] 
    Mp = params[6+2*i+2]*boost[6+2*i+2]
    ind_end = 6+2*i+2 
    if fit_cloud:
        logPtop = params[ind_end+1:ind_end+1+1]*boost[ind_end+1:ind_end+1+1]
        taucloud = fix_taucloud
    else:
        logPtop = [None]
        taucloud = None
    logscale_star, relRV = {}, {}
    for band_tmp in band_unique:
        if band_tmp == 'y':
            if fit_speckle and fit_cloud:
                logscale_star[band_tmp] = fix_logscale_star 
                relRV[band_tmp] = fix_relRV
            elif fit_speckle:
                logscale_star[band_tmp] = fix_logscale_star
                relRV[band_tmp] = fix_relRV
            else:
                logscale_star[band_tmp] = None
                relRV[band_tmp] = None
        elif band_tmp == 'h':
            if fit_speckle and fit_cloud:
                ind_end = ind_end + 1 
                logscale_star[band_tmp] = params[ind_end+1]*boost[ind_end+1]
                relRV[band_tmp] = fix_relRV
            elif fit_speckle:
                logscale_star[band_tmp] = params[ind_end+1]*boost[ind_end+1]
                relRV[band_tmp] = fix_relRV
            else:
                logscale_star[band_tmp] = None
                relRV[band_tmp] = None

    logbeta=[]
    for i in range(len(name_moltel)):
        logbeta.append(
            params[len(params)-len(name_moltel)-len(name_atommol)-len(atommol_unique_p)+i]
            *boost[len(params)-len(name_moltel)-len(name_atommol)-len(atommol_unique_p)+i])

    logvmr=[]
    for i in range(len(name_atommol)):
        logvmr.append(
            params[len(params)-len(name_atommol)-len(atommol_unique_p)+i]
            *boost[len(params)-len(name_atommol)-len(atommol_unique_p)+i])

    u1 = 0.0
    u2 = 0.0

    def normalize_mu(mu,norm):
        nmu = []
        for k in range(len(mu)):
            nmu_k = mu[k]/norm
            nmu.append(nmu_k)
        return nmu

    nmu = []
    transmit = []
    f_obs0 = {}
    for band_tmp in band_unique:
        if band_tmp=='y':
            mu, f_obs0_tmp, transmit_tmp = frun(T0, alpha, logg, logvmr, u1, u2, RV, vsini, 
                                                a[0], b[0], logbeta, vtel, logPtop[0], taucloud, 
                                                ign, obs_grid, band_use='y')
            if norm is None:
                norm0 = f_obs0_tmp
            else:
                norm0 = norm[band_tmp]
            nmu_band = normalize_mu(mu,norm0)
            f_obs0[band_tmp] = f_obs0_tmp
            nmu.extend(nmu_band)
            transmit.extend(transmit_tmp)
        elif band_tmp=='h':
            mu, f_obs0_tmp, transmit_tmp = frun(T0, alpha, logg, logvmr, u1, u2, RV, vsini, 
                                                a[1], b[1], logbeta, vtel, logPtop[0], taucloud, 
                                                ign, obs_grid, band_use='h')
            if norm is None:
                norm0 = f_obs0_tmp
            else:
                norm0 = norm[band_tmp]
            nmu_band = normalize_mu(mu,norm0)
            f_obs0[band_tmp] = f_obs0_tmp
            nmu.extend(nmu_band)
            transmit.extend(transmit_tmp)

    mu = nmu

    if fit_speckle:
        mu_speckle = []
        for band_tmp in band_unique:
            scale_star = 10**logscale_star[band_tmp]
            relRV_tmp = relRV[band_tmp]
            k_use = [i for i,x in enumerate(sum(band,[])) if x==band_tmp]
            f_speckle = scale_speckle(scale_star,relRV_tmp,obs_grid=obs_grid)
            for k in k_use:
                mu_k = (1-scale_star)*mu[k] + f_speckle[k]
                mu_speckle.append(mu_k)
        mu = mu_speckle

    ##Photometry
    logvmr_p=[]
    for j in range(len(name_atommol_p)):
        logvmr_p_j = []
        for i in range(len(name_atommol)):
            if name_atommol[i] == name_atommol_p[j]:
                logvmr_p_j.append(params[len(params)-len(name_atommol)-len(atommol_unique_p)+i]*boost[len(params)-len(name_atommol)+i])
        if len(logvmr_p_j)==0:
            l = atommol_unique_p.index(name_atommol_p[j])
            logvmr_p_j.append(params[len(params)-len(atommol_unique_p)+l]*boost[len(params)-len(atommol_unique_p)+l]) 
        logvmr_p.extend(logvmr_p_j)
    mag = []
    for i in range(num_ord_list_p):
        band_tmp = band_unique[i]
        _, mag_band = frun_p(T0, alpha, logg, Mp, logvmr_p, u1, u2, RV, vsini, logPtop[0], taucloud, band_use=band_tmp)
        mag.append([mag_band])
    return mu, mag, f_obs0, transmit

def objective(params):
    mu, mag, _, _ = model_opt(params, boost)
    f = jnp.concatenate(f_obs)-jnp.concatenate(mu)
    g = jnp.dot(f,f)
    f_mag = jnp.concatenate(jnp.array(mag_obs))-jnp.concatenate(jnp.array(mag))
    g_mag = jnp.dot(f_mag,f_mag) 
    g += g_mag
    return g

# telluric
logbeta = np.ones(len(name_moltel )) * 22.
boost = np.append(boost0, logbeta)
logbeta = np.ones(len(name_moltel )) * 22.
initpar = np.append(initpar0, logbeta)

# molecules
logvmr = np.ones(len(name_atommol)) * (-3.5)
boost = np.append(boost, logvmr)
logvmr = np.ones(len(name_atommol)) * (-3.0)
initpar = np.append(initpar, logvmr)

# photometry
atommol_unique_p = list(set(list(name_atommol))^set(list(name_atommol)+list(name_atommol_p)))
logvmr = np.ones(len(atommol_unique_p)) * (-3.5)
boost = np.append(boost, logvmr)
logvmr = np.ones(len(atommol_unique_p)) * (-3.0)
initpar = np.append(initpar, logvmr)

initpar = initpar / boost


if run:
    """##ADAM
    from jaxopt import OptaxSolver
    import optax
    adam = OptaxSolver(opt=optax.adam(1.e-3), fun=objective)#2.e-2
    params, state = adam.run(init_params=initpar)
    """

    ##TEST
    params = initpar
else:
    params = initpar


#-----------HMC-----------#
from jax import random
import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi
from exojax.utils.gpkernel import gpkernel_RBF

def model_c(nusd,y1,y1err, y2,y2err, ign="ign", obs_grid=True, norm=None):
    T0 = numpyro.sample('T0', dist.Uniform(1200.,3200.))
    alpha = numpyro.sample('alpha', dist.Uniform(0.,1.))
    logg = numpyro.sample('logg', dist.Uniform(4.,6.5))
    Mp = numpyro.sample('Mp', dist.Normal(72.7, 0.8)) 
    RV = numpyro.sample('RV', dist.Uniform(-30.,-10.))
    vsini = numpyro.sample('vsini', dist.Uniform(30.,60.))
    vtel = numpyro.sample('vtel', dist.Uniform(-0.5,0.5))
    for band_tmp in band_unique:
        if band_tmp=='y':
            a_y = numpyro.sample('a_y', dist.Normal(1,0.5))
            b_y = (10**-2)*numpyro.sample('b_y', dist.Uniform(-1,1))
            if fit_cloud:
                logPtop_y = numpyro.sample('logPtop', dist.Uniform(-1.,3.))  
            else:
                logPtop_y = None
            if fit_speckle:
                logscale_star_y = fix_logscale_star 
        elif band_tmp=='h':
            a_h = numpyro.sample('a_h', dist.Normal(1,0.5))
            b_h = (10**-2)*numpyro.sample('b_h', dist.Uniform(-1,1)) 
            if fit_cloud:
                logPtop_h = logPtop_y 
            else:
                logPtop_h = None
            if fit_speckle:
                logscale_star_h = numpyro.sample('logscale_star_h', dist.Uniform(-3.,0.))
    if fit_cloud:
        taucloud = fix_taucloud 
    else:
        taucloud = None
    if fit_speckle:
        relRV = fix_relRV 

    logbeta=[]
    for i in range(len(name_moltel)):
        init_logbeta_tmp = init_dic['logbeta'+name_moltel[i]]
        logbeta.append(numpyro.sample('logbeta'+name_moltel[i], dist.Uniform(15.,25.)))

    logvmr=[]
    for i in range(len(name_atommol)):
        init_logvmr_tmp = init_dic['log'+name_atommol[i]]
        logvmr.append(numpyro.sample('log'+name_atommol[i], dist.Uniform(-15.,0.)))

    u1 = 0.0
    u2 = 0.0

    def normalize_mu(mu,norm):
        nmu = []
        for k in range(len(mu)):
            nmu_k = mu[k]/norm
            nmu.append(nmu_k)
        return nmu

    nmu = []
    transmit = []
    f_obs0 = {}
    for band_tmp in band_unique:
        if band_tmp=='y':
            mu, f_obs0_tmp, transmit_tmp = frun(T0, alpha, logg, logvmr, u1, u2, RV, vsini, 
                                                a_y, b_y, logbeta, vtel, logPtop_y, taucloud, 
                                                ign, obs_grid, band_use='y')
            if norm is None:
                norm0 = f_obs0_tmp
            else:
                norm0 = norm[band_tmp]
            nmu_band = normalize_mu(mu,norm0)
            f_obs0[band_tmp] = f_obs0_tmp
            nmu.extend(nmu_band)
            transmit.extend(transmit_tmp)
        elif band_tmp=='h':
            mu, f_obs0_tmp, transmit_tmp = frun(T0, alpha, logg, logvmr, u1, u2, RV, vsini, 
                                                a_h, b_h, logbeta, vtel, logPtop_h, taucloud, 
                                                ign, obs_grid, band_use='h')
            if norm is None:
                norm0 = f_obs0_tmp
            else:
                norm0 = norm[band_tmp]
            nmu_band = normalize_mu(mu, norm0)
            f_obs0[band_tmp] = f_obs0_tmp
            nmu.extend(nmu_band)
            transmit.extend(transmit_tmp)
    mu = nmu

    if fit_speckle:
        mu_speckle = []
        for band_tmp in band_unique:
            if band_tmp == 'y':
                scale_star = 10**logscale_star_y
            elif band_tmp == 'h':
                scale_star = 10**logscale_star_h
            k_use = [i for i,x in enumerate(sum(band,[])) if x==band_tmp]
            f_speckle = scale_speckle(scale_star, relRV, obs_grid=obs_grid)
            for k in k_use:
                mu_k = (1-scale_star)*mu[k] + f_speckle[k]
                mu_speckle.append(mu_k)
        mu = mu_speckle

    numpyro.deterministic("f_obs0", norm0)


    sigma = numpyro.sample('sigma',dist.Exponential(10.0))
    sig = jnp.ones_like(jnp.concatenate(nusd)) * sigma
    err_all = jnp.sqrt(jnp.concatenate(y1err)**2. + sig**2.)
    if y1 is not None:
        y1 = jnp.concatenate(y1)
    numpyro.sample("y1", dist.Normal(jnp.concatenate(mu), err_all), obs=y1)

    logvmr_p=[]
    for j in range(len(name_atommol_p)):
        logvmr_p_j = []
        for i in range(len(name_atommol)):
            if name_atommol[i] == name_atommol_p[j]:
                logvmr_p_j.append(logvmr[i])
        if len(logvmr_p_j)==0:
            logvmr_p_j.append(numpyro.sample('log'+name_atommol_p[j], dist.Uniform(-15.,0.)))
        logvmr_p.extend(logvmr_p_j)
    mag = []
    for i in range(num_ord_list_p):
        band_tmp = band_unique[i]
        if band_tmp=='y':
            _, mag_band = frun_p(T0, alpha, logg, Mp, logvmr_p, u1, u2, RV, vsini, logPtop_y, taucloud, band_use=band_tmp)
        elif band_tmp=='h':
            _, mag_band = frun_p(T0, alpha, logg, Mp, logvmr_p, u1, u2, RV, vsini, logPtop_h, taucloud, band_use=band_tmp)
        mag.append([mag_band])
    if y2 is not None:
        y2=jnp.concatenate(jnp.array(y2))
    numpyro.sample("y2", dist.Normal(jnp.concatenate(jnp.array(mag)), jnp.concatenate(jnp.array(y2err))), obs=y2)

par_name = ['T0', 'alpha', 'RV', 'vsini', 'vtel']
for band_tmp in band_unique:
    par_name.extend(['a_'+band_tmp,'b_'+band_tmp])
par_name.extend(['logg', 'Mp'])
for band_tmp in band_unique:    
    if fit_cloud:
        if band_tmp=='y':
            par_name.extend(['logPtop'])
    if fit_speckle and band_tmp=='h':
        par_name.extend(['logscale_star_h'])
for i in range(len(name_moltel)):
    par_name.append('logbeta'+name_moltel[i])
for i in range(len(name_atommol)):
    par_name.append('log'+name_atommol[i])
for j in range(len(atommol_unique_p)):
    par_name.append('log'+atommol_unique_p[j])

init = params*boost

from numpyro.infer import init_to_value
init_dic={}
for i,par_name_i in enumerate(par_name):
    init_dic[par_name_i] = init[i]
init_strategy = init_to_value(values=init_dic)

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 2000,200 
kernel = NUTS(model_c, init_strategy=init_strategy)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
if run:
    mcmc.run(rng_key_, nusd=nusd, y1=f_obs, y1err=f_obserr, y2=mag_obs, y2err=magerr_obs)

    mcmc.print_summary()

ord_str = []
for k in range(len(ord_list)):
    ord_str_k = [str(ord) for ord in ord_list[k]]
    ord_str.extend(ord_str_k)
num = '-'.join(ord_str)

import pickle

if run:
    first_samples = mcmc.get_samples()
    with open(save_dir+"/samples_order"+num+"_1.pickle", mode='wb') as f:
        pickle.dump(first_samples, f)
    with open(save_dir+"/samples_order"+num+"_1.pickle", mode='rb') as f:
        samples = pickle.load(f)

    niter = 9 #4
    for i in range(niter):
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, nusd=nusd, y1=f_obs, y1err=f_obserr, y2=mag_obs, y2err=magerr_obs)

        samples_i = mcmc.get_samples()
        with open(save_dir+"/samples_order"+num+f"_{i+2}.pickle", mode='wb') as f:
            pickle.dump(samples_i, f)
        with open(save_dir+"/samples_order"+num+f"_{i+2}.pickle", mode='rb') as f:
            samples = pickle.load(f)

    with open(save_dir+"/mcmc_model_order"+num+".pickle", "wb") as output_file:
        pickle.dump(mcmc, output_file)

    mcmc.print_summary()
else:
    with open(save_dir+"/samples_order"+num+"_1000.pickle", mode='rb') as f:##CHECK!!
        samples = pickle.load(f)
    
    pred = Predictive(model_c,samples,return_sites=["y1","y2"])
    predictions = pred(rng_key_,nusd=nusd,y1=None,y1err=f_obserr,y2=None,y2err=magerr_obs)
    median_mu1 = jnp.median(predictions["y1"],axis=0)
    hpdi_mu1 = hpdi(predictions["y1"], 0.95)
    median_mu2 = jnp.median(predictions["y2"],axis=0)
    hpdi_mu2 = hpdi(predictions["y2"], 0.95)
    len_y = sum([len(ord_list[i]) for i,x in enumerate(sum(band,[])) if x=='y'])
    len_h = sum([len(ord_list[i]) for i,x in enumerate(sum(band,[])) if x=='h'])
    try:
        Ndata_y = int(sum([len(ld_obs[i]) for i,x in enumerate(sum(band,[])) if x=='y'])/len_y)
    except:
        Ndata_y = None
    try:
        Ndata_h = int(sum([len(ld_obs[i]) for i,x in enumerate(sum(band,[])) if x=='h'])/len_h)
    except:
        Ndata_h = None
    median_mu1_all, hpdi_mu1_all = [], []
    ind_end = 0
    for k in range(len(ord_list)):
        if band[k][0]=='y':
            ind_str,ind_end = ind_end, ind_end+Ndata_y*len(ord_list[k])
        elif band[k][0]=='h':
            ind_str, ind_end = ind_end, ind_end+Ndata_h*(len(ord_list[k]))
        median_mu1_all.append(np.array([np.array(median_mu1[ind_str:ind_end])],dtype=object))
        hpdi_mu1_all.append(np.array([np.array(hpdi_mu1[0][ind_str:ind_end]),np.array(hpdi_mu1[1][ind_str:ind_end])],dtype=object))
    median_mu1 = median_mu1_all
    hpdi_mu1 = hpdi_mu1_all
    all_save=[]
    for k in range(len(ord_list)):
        all_save.append(np.array([median_mu1[k],hpdi_mu1[k]],dtype=object))
    all_save = np.array(all_save)
    all_save_mu2=[]
    for k in range(num_ord_list_p):
        all_save_mu2.append(np.array([median_mu2[k],np.array([hpdi_mu2[0][k],hpdi_mu2[1][k]])],dtype=object))
    np.savez(save_dir + "/all_order"+num+".npz",all_save,all_save_mu2)


    par_plot = ['T0','alpha','RV','vsini','vtel']
    for band_tmp in band_unique:
        par_plot.extend(['a_'+band_tmp,'b_'+band_tmp])
        if fit_cloud:
            if band_tmp=='y': # enough to add only one time
                par_plot.extend(['logPtop'])
        if fit_speckle and band_tmp=='h':
            par_plot.extend(['logscale_star_h'])
    for i in range(len(name_moltel)):
        par_plot.append('logbeta'+name_moltel[i])
    for i in range(len(name_atommol)):
        par_plot.append('log'+name_atommol[i])
    for j in range(len(atommol_unique_p)):
        par_plot.append('log'+atommol_unique_p[j])
    par_plot.append('sigma')
    samples_plot = np.array([samples[key] for key in par_plot]).T
    #plot_corner(ord_list, samples_plot,save_dir, labels=par_plot)
    #plot_trace(ord_list, samples, save_dir, num_samples, var_names=(par_plot))


    params_final=[]
    for i,name in enumerate(par_name):
        sample_med = np.median(samples[name])
        params_final.append(sample_med/boost[i])
        if name.startswith('logscale_star'):
            logscale_star_final = np.median(samples[name])

    model_post, mag_post, norm, _ = model_opt(params_final, boost)
    model_wotel, _, _, transmit = model_opt(params_final, boost, ['telluric'], norm=norm)
    model_wo = []
    for i in range(len(name_atommol)):
        model_wo_i, _, _, _ = model_opt(params_final, boost, [name_atommol[i],'telluric'], norm=norm)
        model_wo.append(model_wo_i)


    f_speckle = []
    for band_tmp in band_unique:
        if band_tmp == 'y':
            scale_star = 10**fix_logscale_star
        elif band_tmp == 'h':
            scale_star = 10**logscale_star_final
        relRV = fix_relRV
        k_use = [i for i,x in enumerate(sum(band,[])) if x==band_tmp]
        f_speckle_tmp = scale_speckle(scale_star,relRV,obs_grid=True)
        for k in k_use:
            f_speckle_k = f_speckle_tmp[k]
            f_speckle.append(f_speckle_k)

    save_file = save_dir + "/models_order"+num+".npz"
    all_save=[]
    for k in range(len(ord_list)):
        all_save_k = [name_atommol_masked[k], ord_list[k], ld_obs[k], f_obs[k],
                    f_speckle[k], model_wotel[k], transmit[k]]
        model_wo_k = []
        for i in range(len(name_atommol_masked[k])):
            model_wo_k.append(model_wo[i][k])
        all_save_k = [all_save_k, model_wo_k]
        all_save_k = np.array(all_save_k, dtype=object)
        all_save.append(all_save_k)
    all_save = np.array(all_save)
    np.savez(save_file, all_save)

te = time.time()
print('time = ',te-ts)

print("end main_hmc.py")
       
