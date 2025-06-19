import numpy as np

def reduced_chisquare(data, model, error, sigma, dof):
    chisq = np.sum(
                (np.concatenate(data) - np.concatenate(model))**2 
                / (np.concatenate(error)**2 + sigma**2)
                )
    return chisq, chisq / dof

def bayesian_information_criterion(loglike, n_params, n_data):
    bic = -2 * loglike + n_params * np.log(n_data)
    return bic

if __name__ == '__main__':
    from pathlib import Path
    import os
    import pickle
    import obs
    import setting
    from plotutils import load_data_all, load_data_model, load_params_mle

    path_obs, path_data, path_telluric, path_save = setting.set_path()

    ### SETTINGS ###
    target = "HR7672B"
    date = "20210624"
    mmf = "m2"
    order = [43, 44, 45, 57, 58, 59, 60]

    ord_list = [[x] for x in order]
    ord_str = []
    for k in range(len(ord_list)):
        ord_str_k = [str(ord) for ord in ord_list[k]]
        ord_str.extend(ord_str_k)
    num = '-'.join(ord_str)

    output_dir = Path(f"/home/yuikasagi/Develop/exojax/output/multimol/{target}/{date}/hmc_unilogg/")

    file_samples = output_dir / f"samples_order{num}_1000.pickle"
    file_all = output_dir / f"all_order{num}.npz"
    #file_model = output_dir / f"models_map_order{num}.npz"
    file_mle = output_dir / f"params_mle_order{num}.npz"

    print("sample file: ", str(file_samples))
    ###############

    band=[]
    for k in range(len(ord_list)):
        if ord_list[k][0]<52:
            band.append(['y'])
        elif ord_list[k][0]>51:
            band.append(['h'])
    band_unique = sorted(set(sum(band,[])))[::-1] ## sort y,h

    ord_norm = {}
    for k in range(len(ord_list)):
        if band[k][0] == 'y':
            ord_norm["y"] = 44 
        elif band[k][0] == 'h':
            ord_norm["h"] = 59 

    # samples
    with open(file_samples, mode='rb') as f:
        samples = pickle.load(f)

    ign_keys = ["f_obs0"]
    params = []
    for k, v in samples.items():
        if k in ign_keys:
            continue
        else:
            params.append(k)

    # prediction, model, MLE
    data_all = np.load(file_all, allow_pickle=True)
    #data_model = np.load(file_model, allow_pickle=True)
    params_mle = np.load(file_mle, allow_pickle=True)

    # setting parameter name
    if "logPtop" in samples.keys():
        par_name = ['T0', 'alpha', 'RV', 'vsini', 'vtel', 'a_y', 'b_y', 'a_h', 'b_h', 'logg', 'Mp', 'logPtop', 'logscale_star_h', 'logbetaH2O', 'logbetaCH4', 'logbetaCO2', 'logbetaO2', 'logH2O', 'logFeH']
    else:
        par_name = ['T0', 'alpha', 'RV', 'vsini', 'vtel', 'a_y', 'b_y', 'a_h', 'b_h', 'logg', 'Mp', 'logscale_star_h', 'logbetaH2O', 'logbetaCH4', 'logbetaCO2', 'logbetaO2', 'logH2O', 'logFeH']

    flux_median_mu, flux_hpdi_mu, mag_median_mu, mag_hpdi_mu = load_data_all(data_all, order)
    #name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_post, model_wotel, transmitA, model_wo, mag_post = load_data_model(data_model, order)
    maxloglike, mle_params = load_params_mle(params_mle)
    print("maximum loglikelihood = ",maxloglike)
    print("MLE = ",dict(zip(par_name, mle_params)))

    # observed spectrum
    path_spec = {}
    for band_tmp in band_unique:
        path_spec[band_tmp] = os.path.join(path_obs,f'hr7672b/nw{target}_{date}_{band_tmp}_{mmf}_photnoise.dat') 

    ld_obs,f_obs,f_obserr = [], [], []
    ld0 = {}
    for band_tmp in band_unique:
        ind_tmp = [x==band_tmp for x in sum(band,[])]
        ld_obs_tmp, f_obs_tmp, f_obserr_tmp, ord_tmp, ld0_tmp = obs.spec(
            path_spec[band_tmp], path_telluric, band_tmp, np.array(ord_list)[ind_tmp], ord_norm=ord_norm[band_tmp], 
            norm=False)
        
        ld_obs.extend(ld_obs_tmp)
        f_obs.extend(f_obs_tmp)
        f_obserr.extend(f_obserr_tmp)
        ld0[band_tmp] = ld0_tmp

    # numbers
    n_params = len(params)
    n_data = len(np.concatenate(f_obs))
    dof = n_data - n_params

    sigma = np.median(samples["sigma"])
    print(sigma)
    
    # reduced chi square
    chisq, red_chisq = reduced_chisquare(f_obs, flux_median_mu, f_obserr, sigma, dof)
    print("chi-square = ",chisq)
    print("reduced chi-square = ",red_chisq)

    # MLE 
    loglike = maxloglike
    bic = bayesian_information_criterion(loglike, n_params, n_data)
    print("BIC = ", bic)

    