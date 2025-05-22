def load_data_all(data_all, order):
    flux_median_mu, flux_hpdi_mu = [], [] 
    for k in range(len(order)):
        flux_median_mu_k, flux_hpdi_mu_k = data_all['arr_0'][k][0][0], data_all['arr_0'][k][1].astype(float)
        flux_median_mu.append(flux_median_mu_k)
        flux_hpdi_mu.append(flux_hpdi_mu_k)

    mag_median_mu, mag_hpdi_mu = data_all['arr_1'][0][0], data_all['arr_1'][0][1]
    return flux_median_mu, flux_hpdi_mu, mag_median_mu, mag_hpdi_mu

def load_data_model(data_model, order):
    models = data_model['arr_0']
    mag_post = data_model['arr_1']
    name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_post, model_wotel, transmitA = [], [], [], [], [], [], [], []
    model_wo_i0, model_wo_i1 = [], []
    for k in range(len(order))[:]:
        name_atommol_masked_k, ord_list_k, ld_obs_k, f_obs_k, f_speckle_k, model_post_k, model_wotel_k, transmitA_k = models[k][0]
        #name_atommol_masked_k, ord_list_k, ld_obs_k, f_obs_k, f_speckle_k, model_wotel_k = models[k][0]
        model_wo_k = models[k][1]
        name_atommol_masked.append(name_atommol_masked_k)
        ord_list.append(ord_list_k)
        ld_obs.append(ld_obs_k)
        f_obs.append(f_obs_k)
        f_speckle.append(f_speckle_k)
        model_post.append(model_post_k)
        model_wotel.append(model_wotel_k)
        transmitA.append(transmitA_k)
        if len(model_wo_k) == 1:
            model_wo_i0.append(model_wo_k[0])
        elif len(model_wo_k)==2:
            model_wo_i0.append(model_wo_k[0])
            model_wo_i1.append(model_wo_k[1])
    model_wo = [model_wo_i0, model_wo_i1]
    return name_atommol_masked, ord_list, ld_obs, f_obs, f_speckle, model_post, model_wotel, transmitA, model_wo, mag_post

def load_params_mle(params_mle):
    maxloglike = params_mle['arr_0']
    params = params_mle['arr_1']
    return maxloglike, params
