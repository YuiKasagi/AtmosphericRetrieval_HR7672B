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
sns.set_context("notebook")
# %%
#-----------Settings-----------#
output_dir = Path("/home/yuikasagi/Develop/exojax/output/multimol/HR7672B/20210624/hmc_wocloud_ulogg_nm/")

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
## read data of data_all
flux_median_mu, flux_hpdi_mu = [], [] 
for k in range(len(order)):
    flux_median_mu_k, flux_hpdi_mu_k = data_all['arr_0'][k][0][0], data_all['arr_0'][k][1].astype(float)
    flux_median_mu.append(flux_median_mu_k)
    flux_hpdi_mu.append(flux_hpdi_mu_k)

mag_median_mu, mag_hpdi_mu = data_all['arr_1'][0][0], data_all['arr_1'][0][1]
# %%
print("Jmag (pred)= %.2f, [%.2f,%.2f]"%(mag_median_mu, mag_hpdi_mu[0],mag_hpdi_mu[1]))

# %%
## read data of data_model
models = data_model['arr_0']
name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_wotel, transmitA = [], [], [], [], [], [], []
model_wo_i0, model_wo_i1 = [], []
for k in range(len(order))[:]:
    name_atommol_masked_k, ord_list_k, ld_obs_k, f_obs_k, f_speckle_k, model_wotel_k, transmitA_k = models[k][0]
    #name_atommol_masked_k, ord_list_k, ld_obs_k, f_obs_k, f_speckle_k, model_wotel_k = models[k][0]
    model_wo_k = models[k][1]
    name_atommol_masked.append(name_atommol_masked_k)
    ord_list.append(ord_list_k)
    ld_obs.append(ld_obs_k)
    f_obs.append(f_obs_k)
    f_speckle.append(f_speckle_k)
    model_wotel.append(model_wotel_k)
    transmitA.append(transmitA_k)
    if len(model_wo_k) == 1:
        model_wo_i0.append(model_wo_k[0])
    elif len(model_wo_k)==2:
        model_wo_i0.append(model_wo_k[0])
        model_wo_i1.append(model_wo_k[1])
model_wo = [model_wo_i0, model_wo_i1]



# %%
import sys
sys.path.insert(0,'/home/yuikasagi/exojax/src')

from exojax.utils.mollabel import format_molecules_lists
from matplotlib.ticker import AutoMinorLocator

def plot_speckle(model_post, model_hpdi, model_wotel, model_wo, name_atommol_masked, 
                ord_list, ld_obs, f_obs, f_speckle, save_dir, **kwargs):
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
        fig,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(20,12),gridspec_kw={'height_ratios': [4, 3, 3, 2]},sharex=True)
        plt.subplots_adjust(hspace=0.)
        labels = ["Corrected Data","BD Model"]
        mol_labels, handles = [], []
        for k in k_use_tmp:
            ax1.plot(ld_obs[k],f_obs[k],'k.',ms=3,zorder=0)
            ax1.plot(ld_obs[k],model_post[k],color='tab:blue',zorder=2)
            ax1.fill_between(ld_obs[k],model_hpdi[k][0],model_hpdi[k][1],color='tab:blue',alpha=0.2,zorder=2)

            ax2.plot(ld_obs[k],transmit[k] ,color='skyblue',lw=1.5,alpha=0.5,zorder=1)

            #f_speckle_k = f_speckle[k]
            f_speckle_k = f_speckle[k] / transmit[k]
            ax2.plot(ld_obs[k],f_speckle_k,color='tab:purple',alpha=0.5,lw=1.5,zorder=1)#,alpha=0.5)

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
        ax2.legend(['Transmittance','Speckle','BD Model'],ncol=3)
        labels[1:1] = mol_labels
        ax3.legend(handles,labels,ncol=2)#,loc='upper right')
        ax1.set(xlim=(min(ld_obs[k_use_tmp[0]]),max(ld_obs[k_use_tmp[-1]])),ylim=(0.,1.6))
        ax1.set(ylabel='normalized flux')
        ax2.set(ylabel='normalized flux',ylim=(0,1.2))
        #ax3.set(title="Corrected Data and BD Model")
        ax3.set(ylim=(0.,1.2),ylabel='normalized flux')
        #ax3.set(xlabel='wavelength [$\AA$]')
        ax4.set(ylabel="residuals",xlabel='wavelength [$\AA$]',ylim=(-0.4,0.4))

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
                 ord_list, ld_obs, f_obs, f_speckle, save_dir, band=band)
