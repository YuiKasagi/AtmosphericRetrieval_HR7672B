# %%
"""Plot HMC results from main_hmc.py"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# %%
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
# %%
#-----------Settings-----------#
output_dir = Path("/home/yuikasagi/Develop/exojax/output/multimol/HR7672B/20210624/hmc_ulogg_nm/")

Ttop = 2112.
RV = -14.62# retrieved value
mjd_B = 59390.46496304 # for barycentric correction
mjd_A = 59372.64163201
ra_deg = 301.0259167
dec_deg = 17.0701861

order = [43, 44, 45, 57, 58, 59, 60]

ord_list = [[x] for x in order]
band=[]
for k in range(len(ord_list)):
    if ord_list[k][0]<52:
        band.append(['y'])
    elif ord_list[k][0]>51:
        band.append(['h'])

## load data
order_connect = '-'.join([str(x) for x in order])
file_all = output_dir / f"all_order{order_connect}.npz"
file_model = output_dir / f"models_order{order_connect}.npz"

data_all = np.load(file_all,allow_pickle=True)
data_model = np.load(file_model,allow_pickle=True)
#----------------------#

# %%
from plotutils import load_data_all, load_data_model
## read data of data_all
flux_median_mu, flux_hpdi_mu, mag_median_mu, mag_hpdi_mu = load_data_all(data_all, order)
# %%
print("Jmag (pred)= %.2f, [%.2f,%.2f]"%(mag_median_mu, mag_hpdi_mu[0],mag_hpdi_mu[1]))

# %%
## read data of data_model
name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_post, model_wotel, transmitA, model_wo, mag_post = load_data_model(data_model, order, mle=False)

# %%
# molecular database
from exojax.spec.api import MdbExomol
from exojax.utils.grids import wavenumber_grid  
import numpy as np
import math

path_data = "/home/yuikasagi/Develop/exojax/database/"

wavmin = np.min(ld_obs[0])
wavmax = np.max(ld_obs[-1])
R = 1000000.

nu_min = 1.0e8/(wavmax+5.0)
nu_max = 1.0e8/(wavmin-5.0)
Nx = math.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki
Nx = math.ceil(Nx/2.) * 2 # make even
nu_grid, wav, res = wavenumber_grid(wavmin-5, wavmax+5, Nx, "premodit", unit="AA") 

mdb_H2O = MdbExomol(path_data + "H2O/1H2-16O/POKAZATEL", nurange=nu_grid, activation=False)
mdb_FeH = MdbExomol(path_data + "FeH/56Fe-1H/MoLLIST", nurange=nu_grid, activation=False)
# %%
mask_y = [k[0]=='y' for k in band]
mask_h = [k[0]=='h' for k in band]
ld_obs_y = [ld_obs[k] for k in range(len(ld_obs)) if mask_y[k]]
ld_obs_h = [ld_obs[k] for k in range(len(ld_obs)) if mask_h[k]]

def mdb_to_df(mdb, Teff=2100., Sij_threshold=1e-19):
    df = mdb.df #.to_pandas_df()
    mask = mdb.df_load_mask
    df_mask = df[mask]
    mdb.attributes_from_dataframes(df_mask)
    Sij = mdb.line_strength(Teff) # exojax2
    df_mask["Sij"] = Sij
    df_y = df_mask[df_mask["nu_lines"].between(1e8/np.max(ld_obs_y), 1e8/np.min(ld_obs_y))]
    df_h = df_mask[df_mask["nu_lines"].between(1e8/np.max(ld_obs_h), 1e8/np.min(ld_obs_h))]
    #df_y = df_y.sort_values("Sij0", ascending=False)[:N_head]
    #df_h = df_h.sort_values("Sij0", ascending=False)[:N_head]
    df_y = df_y[df_y["Sij"] > Sij_threshold]
    df_h = df_h[df_h["Sij"] > Sij_threshold]
    return df_mask, df_y, df_h

df_H2O, df_H2O_y, df_H2O_h = mdb_to_df(mdb_H2O, Teff=Ttop, Sij_threshold=5e-23)#6e-23)
df_FeH, df_FeH_y, df_FeH_h = mdb_to_df(mdb_FeH, Teff=Ttop, Sij_threshold=3e-19)
# %%
# RV & barycentric correction
from obs import barycentric_correction
import exojax.utils.constants as const
barycorr_B = barycentric_correction(ra_deg, dec_deg, mjd_B) # km/s
barycorr_A = barycentric_correction(ra_deg, dec_deg, mjd_A) # km/s
ld_obs_B = [ld_obs[k] * (1.0 - (RV + barycorr_B)/const.c) for k in range(len(ld_obs))] 
ld_obs_A = [ld_obs[k] * (1.0 - (RV + barycorr_A)/const.c) for k in range(len(ld_obs))]
  
# %%
#import sys
#sys.path.insert(0,'/home/yuikasagi/exojax/src')

from exojax.utils.mollabel import format_molecules_lists
from matplotlib.ticker import AutoMinorLocator

def plot_speckle(model_post, model_hpdi, model_wotel, model_wo, name_atommol_masked, 
                ord_list, ld_obs, ld_obs_A, f_obs, f_speckle, save_dir, **kwargs):
    """plot models

    Args:
        model_post: posterior median of the model
        model_hpdi: 95% HPDI of the model
        model_wotel: model without tellurics
        model_wo: model without molecules
        name_atommol_masked: masked molecules
        ord_list: orders
        ld_obs: observed wavelength
        f_obs: observed flux
        f_speckle: speckle noise
        save_dir: directory to save the plot
        kwargs: band
    """
    ord_str = []
    for k in range(len(ord_list)):
        ord_str_k = [str(ord) for ord in ord_list[k]]
        ord_str.extend(ord_str_k)
    num0 = '-'.join(ord_str)

    color = [["C%d"%(i+1) for i in range(max([len(x) for x in name_atommol_masked]))]]

    #labels = ["corrected data","BD model"]
    #mol_labels = ["w/o "+x for x in mols_name[0]]
    #labels[1:1] = mol_labels

    if len(kwargs.keys())!=0:
        band = kwargs['band']
        band_unique = sorted(set(sum(band,[])))[::-1] ## sort y,h
        k_use = []
        for band_tmp in band_unique:
            k_use.append([i for i,x in enumerate(sum(band,[])) if x==band_tmp])
        transmit = []
        for k in range(len(ord_list)):
            #transmit.append(model_post[k].astype(float)/np.array(model_wotel[k])) #for median_mu1
            transmit.append((model_post[k].astype(float)-f_speckle[k])/np.array(model_wotel[k]-f_speckle[k])) #for median_mu1
    else:
        k_use = [range(len(ord_list))]
        transmit = model_post/model_wotel

    for j,k_use_tmp in enumerate(k_use):
        print(k_use_tmp)
        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(20,15),gridspec_kw={'height_ratios': [16, 13, 12, 6]},sharex=True)
        plt.subplots_adjust(hspace=0.)
        labels = ["Corrected Data","BD Model"]
        mol_labels, handles = [], []
        for k in k_use_tmp:
            ax1.plot(ld_obs[k],f_obs[k],'k.',ms=3,zorder=0)
            ax1.plot(ld_obs[k],model_post[k],color='tab:blue',zorder=2)
            ax1.fill_between(ld_obs[k],model_hpdi[k][0],model_hpdi[k][1],color='tab:blue',alpha=0.2,zorder=2)

            ax2.plot(ld_obs[k],transmit[k] ,color='grey',lw=1.5,alpha=0.5,zorder=1)

            #f_speckle_k = f_speckle[k]
            f_speckle_k = f_speckle[k] / transmit[k]
            ax2.plot(ld_obs_A[k],f_speckle_k,color='black',alpha=0.5,lw=1.5,zorder=1)
            if band[k][0]=='y':
                ax2.vlines(1e8/df_H2O_y["nu_lines"].values, ymin=-0.1, ymax=0.05, lw=1.5, color="tab:orange", alpha=0.2)
                ax2.vlines(1e8/df_FeH_y["nu_lines"].values, ymin=0.05, ymax=0.2, lw=1.5, color="tab:green", alpha=0.5)
            if band[k][0]=='h':
                ax2.vlines(1e8/df_H2O_h["nu_lines"].values, ymin=-0.1, ymax=0.05, lw=1.5, color="tab:orange", alpha=0.2)

            corrected_spec_k = (f_obs[k] - f_speckle[k])/transmit[k]
            line, = ax3.plot(ld_obs[k],corrected_spec_k,'k.',ms=3)
            if k==k_use_tmp[0]:
                handles.append(line)
            for i in range(len(name_atommol_masked[k])):
                if len(model_wo[i][k])==0:
                    ax3.plot(0,0,color=color[0][i])
                else:
                    line, = ax3.plot(ld_obs[k],model_wo[i][k]-f_speckle[k],color=color[0][i],alpha=0.8,lw=2.5)
                    if k==k_use_tmp[0]:
                        mols_name_k = format_molecules_lists([name_atommol_masked[k]])
                        handles.append(line)
                        mol_labels.append("w/o "+mols_name_k[0][i])#name_atommol_masked[k][i])
            model_BD_k = model_wotel[k]-f_speckle[k]
            ax2.plot(ld_obs[k],model_BD_k,color='hotpink',lw=2.5,zorder=2)#,alpha=0.8)
            line, = ax3.plot(ld_obs[k],model_BD_k,color='hotpink',lw=2.5,alpha=0.8)
            if k==k_use_tmp[0]:
                handles.append(line)
            #ax4.fill_between(ld_obs[k],model_wotel[k]+hpdi_y1_gp[k][0],model_wotel[k]+hpdi_y1_gp[k][1],alpha=0.2,color='hotpink')

            try:
                ax4.plot(ld_obs[k],f_obs[k]-(model_post[k]),'.',color='grey',alpha=0.5,ms=3)
            except:
                ax4.plot(ld_obs[k],f_obs[k]-(model_post[k][0]),'.',color='grey',alpha=0.5,ms=3) #for median_mu1

        ax1.legend(['Observed Data','Full Model','95% area'],ncol=3)#,'Speckle','Transmittance','BD Model','Residuals'],ncol=4,bbox_to_anchor=(0.5, 1.3), loc='upper center')
        #ax2.legend(['Transmittance','Speckle','BD Model'],ncol=3)
        if band[k][0]=='y':
            ax2.text(0.85, 0.9, "Transmittance", transform=ax2.transAxes)
            ax2.text(0.85, 0.7, "Speckle", transform=ax2.transAxes)
            ax2.text(0.85, 0.41, "BD Model", transform=ax2.transAxes)
            ax2.text(0.85, 0.08, "FeH", transform=ax2.transAxes, color="tab:green", fontweight="bold")
        if band[k][0]=='h':
            ax2.text(0.85, 0.9, "Transmittance", transform=ax2.transAxes)
            ax2.text(0.85, 0.71, "BD Model", transform=ax2.transAxes)
            ax2.text(0.85, 0.43, "Speckle", transform=ax2.transAxes)
            ax2.text(0.85, 0.19, "H$_2$O", transform=ax2.transAxes, color="tab:orange", fontweight="bold")
        labels[1:1] = mol_labels
        ax3.legend(handles,labels,ncol=2)#,loc='upper right')
        ax1.set(xlim=(min(ld_obs[k_use_tmp[0]]),max(ld_obs[k_use_tmp[-1]])),ylim=(0.,1.6))
        ax1.set(ylabel='normalized flux')
        ax2.set(ylabel='normalized flux',ylim=(-0.15,1.15))
        #ax3.set(title="Corrected Data and BD Model")
        ax3.set(ylim=(0.,1.2),ylabel='normalized flux')
        #ax3.set(xlabel='wavelength [$\AA$]')
        ax4.set(ylabel="residuals",xlabel='wavelength [$\AA$]',ylim=(-0.3,0.3))

        for ax in [ax1,ax2,ax3,ax4]: 
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        #ax2.patch.set_alpha(0)
        ax1.tick_params(labelbottom=False, labeltop=True)

        if len(kwargs.keys())!=0:
            num = num0+'_'+band_unique[j]
        else:
            num = num0

        plt.savefig(save_dir+"/hmc_speckle_order"+num+".png", bbox_inches='tight',dpi=300)
        #plt.show()

# %%
save_dir = str(output_dir)
plot_speckle(flux_median_mu, flux_hpdi_mu, model_wotel, model_wo, name_atommol_masked, 
                 ord_list, ld_obs_B, ld_obs_A, f_obs, f_speckle, save_dir, band=band)
