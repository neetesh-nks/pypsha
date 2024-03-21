# -*- coding: utf-8 -*-
"""
Jan 2023

@author: neetesh
"""

import numpy as np
import copy
import cvxpy as cp

from itertools import combinations
from scipy.stats import lognorm, poisson
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.stats import weightstats
from scipy.interpolate import interp1d

def haversine_distance(lon1,lat1,lon2,lat2,radius = 6371):
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    dangle = 2 * np.arcsin(np.sqrt( np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2 ))
    
    return radius*dangle

def probability_of_exceedance(im,event_set,intensity_id,sitei):
    rates = event_set.events.metadata.loc[:,['annualized_rate']].values.flatten()
    data_df = event_set.events.data.loc[intensity_id,[sitei]]
    sigmas = data_df.loc['sigma',:].values.flatten()
    mus = data_df.loc['mu',:].values.flatten()
    probs = 1 - lognorm.cdf(im, s=sigmas, scale= np.exp(mus))
    return 1 - poisson.pmf(k=0,mu=(probs*rates).sum())

def meter_to_unitlatlong(minlat,maxlat):
    meter2ddlat = 111111
    meter2ddlong = np.cos(np.mean((minlat,maxlat))*np.pi/180)*111111
    return (meter2ddlat,meter2ddlong)

def hazard_curves_from_weights(imdf,weights,metadata,imvals = np.logspace(-3,1,100)):
    metadata = copy.deepcopy(metadata)
    metadata['w'] = weights
    imvals_broad = np.repeat(imvals.reshape((-1,1)),imdf.shape[1],axis=1)
    rate_exceedance = np.zeros((imvals.shape[0],imdf.shape[1]))
    for source,rupture in metadata.index:
        imdf_event = imdf.loc[source,rupture,:]
        rate_exceedance+=imdf_event.apply(lambda x : 1 - ECDF(x)(imvals),axis=0).values*metadata.loc[source,rupture].w
    return (imvals_broad,rate_exceedance)

def clean_renormalize_weights(w0,w1,k=None,eps=1e-10):
    w1[w1<eps] = 0
    w1 = (w1/w1.sum())*w0.sum()
    if not k is None:
        k_highest_index = np.argpartition(w1, -k)[-k:]
        w2 = np.zeros_like(w1)
        w2[k_highest_index] = w1[k_highest_index]
        w2 = (w2/w2.sum())*w0.sum()
        return w2
    else:
        return w1

def optimize_maps(imdf,w0,im,hz_curves,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_site_indices=None,
                     weight_exponent = -1.0,
                     site_weights = None,
                     omega = None,
                     verbose=True,
                     solver = 'ECOS'):
    
    imdf = copy.deepcopy(imdf)
    nsites = imdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    site_curves_mins = np.maximum(hz_curves.min(axis=0),10**hz_curves_min)
    site_curves_maxs = np.minimum(hz_curves.max(axis=0),10**hz_curves_max)
    
    
    curve_points = np.array([np.geomspace(site_curves_mins[sitei], site_curves_maxs[sitei],n_curve_points) for sitei in range(nsites)]).T
    
    ims = np.array([interp1d(hz_curves[:,i],im[:,i])(curve_points[:,i]) for i in range(nsites)])
    
    if candies_per_event is None:
        imcandy = imdf
    else:
        imcandy = imdf.loc[:,:,:candies_per_event-1]
        
    if not candies_site_indices is None:
        imcandy = imcandy.iloc[:,candies_site_indices]
        ims = ims[candies_site_indices,:]
        nsites = candies_site_indices.shape[0]

    imcandy_ar = imcandy.values
    
    
    theta = np.zeros((nsites*curve_points.shape[0],imcandy.shape[0]))
    for i in range(nsites):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    site_lambdas = np.tile(base_curve_points**weight_exponent,nsites)
    if not site_weights is None:
        site_weights = np.repeat(site_weights, n_curve_points)
        site_lambdas = site_weights*site_lambdas
    lambdas_IR = np.diag(site_lambdas)

    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    
    if omega is None:
        omega = w0.sum()
    objective = cp.Minimize(cp.norm(1 - lambdas_IR@U@x,1))
    constraints = [x>=0,cp.sum(x)<=omega]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}


def optimize_maps_milp(imdf,w0,im,hz_curves,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_site_indices=None,
                     weight_exponent = -1.0,
                     site_weights = None,
                     k = 20,
                     verbose=True,
                     solver = 'SCIP',
                     TimeLimit = 600):
    
    imdf = copy.deepcopy(imdf)
    nsites = imdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    site_curves_mins = np.maximum(hz_curves.min(axis=0),10**hz_curves_min)
    site_curves_maxs = np.minimum(hz_curves.max(axis=0),10**hz_curves_max)
    
    
    curve_points = np.array([np.geomspace(site_curves_mins[sitei], site_curves_maxs[sitei],n_curve_points) for sitei in range(nsites)]).T
    
    ims = np.array([interp1d(hz_curves[:,i],im[:,i])(curve_points[:,i]) for i in range(nsites)])
    
    if candies_per_event is None:
        imcandy = imdf
    else:
        imcandy = imdf.loc[:,:,:candies_per_event-1]
        
    if not candies_site_indices is None:
        imcandy = imcandy.iloc[:,candies_site_indices]
        ims = ims[candies_site_indices,:]
        nsites = candies_site_indices.shape[0]

    imcandy_ar = imcandy.values
    
    
    theta = np.zeros((nsites*curve_points.shape[0],imcandy.shape[0]))
    for i in range(nsites):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    site_lambdas = np.tile(base_curve_points**weight_exponent,nsites)
    if not site_weights is None:
        site_weights = np.repeat(site_weights, n_curve_points)
        site_lambdas = site_weights*site_lambdas
    lambdas_IR = np.diag(site_lambdas)
    curve_point_matrix = curve_points.T.flatten()

    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    z = cp.Variable(N, boolean = True)
    # if omega is None:
    #     omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1))
    constraints = [x>=0,cp.sum(z)<=k,x<=z]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose,scip_params = {"limits/time" : TimeLimit},TimeLimit=TimeLimit)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def optimize_maps_loss_milp(imdf,w0,im,hz_curves,
                     lossdf, loss_im,loss_hz_curves,
                     alpha = 0.5,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_site_indices=None,
                     weight_exponent = -1.0,
                     site_weights = None,
                     k = 20,
                     verbose=True,
                     solver = 'SCIP',
                     TimeLimit = 600):
    
    imdf = copy.deepcopy(imdf)
    nsites = imdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    site_curves_mins = np.maximum(hz_curves.min(axis=0),10**hz_curves_min)
    site_curves_maxs = np.minimum(hz_curves.max(axis=0),10**hz_curves_max)
    
    lossdf = copy.deepcopy(lossdf)
    nproxies = lossdf.shape[1]
    proxy_curves_mins = np.maximum(loss_hz_curves.min(axis=0),10**hz_curves_min)
    proxy_curves_maxs = np.minimum(loss_hz_curves.max(axis=0),10**hz_curves_max)
    
    curve_points = np.array([np.geomspace(site_curves_mins[sitei], site_curves_maxs[sitei],n_curve_points) for sitei in range(nsites)]).T
    loss_curve_points = np.array([np.geomspace(proxy_curves_mins[lossi], proxy_curves_maxs[lossi],n_curve_points) for lossi in range(nproxies)]).T
    
    ims = np.array([interp1d(hz_curves[:,i],im[:,i])(curve_points[:,i]) for i in range(nsites)])
    
    proxies = np.array([interp1d(loss_hz_curves[:,i],loss_im[:,i])(loss_curve_points[:,i]) for i in range(nproxies)])

    if candies_per_event is None:
        imcandy = imdf
        losscandy = lossdf
    else:
        imcandy = imdf.loc[:,:,:candies_per_event-1]
        losscandy = lossdf.loc[:,:,:candies_per_event-1]
        
    if not candies_site_indices is None:
        imcandy = imcandy.iloc[:,candies_site_indices]
        ims = ims[candies_site_indices,:]
        nsites = candies_site_indices.shape[0]

    imcandy_ar = imcandy.values
    losscandy_ar = losscandy.values
        
    theta = np.zeros((nsites*curve_points.shape[0],imcandy.shape[0]))
    for i in range(nsites):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1

    losstheta = np.zeros((nproxies*loss_curve_points.shape[0],losscandy.shape[0]))
    for i in range(nproxies):
        losstheta[i*loss_curve_points.shape[0]:(i+1)*loss_curve_points.shape[0],:] = (proxies[i:i+1,:] < losscandy_ar[:,i:i+1]).T*1
    
    base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    
    site_lambdas = np.tile(base_curve_points**weight_exponent,nsites)
    if not site_weights is None:
        site_weights = np.repeat(site_weights, n_curve_points)
        site_lambdas = site_weights*site_lambdas
    lambdas_IR = np.diag(site_lambdas)
    curve_point_matrix = curve_points.T.flatten()

    loss_lambdas = np.tile(base_curve_points**weight_exponent,nproxies)
    loss_lambdas_IR = np.diag(loss_lambdas)
    loss_curve_point_matrix = loss_curve_points.T.flatten()

    
    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    z = cp.Variable(N, boolean = True)

    Uloss = losstheta
    
    # if omega is None:
    #     omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1) +
                            alpha*cp.norm(loss_lambdas_IR@(loss_curve_point_matrix - Uloss@x),1))
    constraints = [x>=0,cp.sum(z)<=k,x<=z]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose,scip_params = {"limits/time" : TimeLimit},TimeLimit=TimeLimit)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'loss_df':losscandy,
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def optimize_maps_pca(pcdf,w0,pcavals_broad,pca_rate_exceedance, pca_wt_params,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_comp_indices=None,
                     weight_exponent = -1.0,
                     comp_weights = None,
                     omega = None,
                     verbose=True,
                     solver = 'ECOS'):
    
    pcdf = copy.deepcopy(pcdf)
    ncomps = pcdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    comp_curves_mins = np.maximum(pca_rate_exceedance.min(axis=0),10**hz_curves_min)
    comp_curves_maxs = np.minimum(pca_rate_exceedance.max(axis=0),10**hz_curves_max)
    
    curve_points = np.array([np.geomspace(comp_curves_mins[compi], comp_curves_maxs[compi],n_curve_points) for compi in range(ncomps)]).T
    
    pca_prob_exceedance = 1-poisson.pmf(k=0,mu=pca_rate_exceedance)
    pca_prob_exceedance = pca_prob_exceedance/pca_prob_exceedance.max()
    symmetric_quants = np.linspace(0.005,1-0.005,n_curve_points)
    im_symmetric_quants = np.array([interp1d(pca_prob_exceedance[:,i],pcavals_broad[:,i])(symmetric_quants) for i in range(ncomps)]).T
    comp_curve_points = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(im_symmetric_quants[:,i]) for i in range(ncomps)]).T
    curve_points[:,1:] = comp_curve_points[:,1:]
    
    ims = np.array([interp1d(pca_rate_exceedance[:,i],pcavals_broad[:,i])(curve_points[:,i]) for i in range(ncomps)])
    
    if candies_per_event is None:
        imcandy = pcdf
    else:
        imcandy = pcdf.loc[:,:,:candies_per_event-1]
        
    if not candies_comp_indices is None:
        imcandy = imcandy.iloc[:,candies_comp_indices]
        ims = ims[candies_comp_indices,:]
        ncomps = candies_comp_indices.shape[0]

    imcandy_ar = imcandy.values
    
    theta = np.zeros((ncomps*curve_points.shape[0],imcandy.shape[0]))
    for i in range(ncomps):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    # base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    comp_lambdas = curve_points**weight_exponent
    for i in range(1,ncomps):
        comp_lambdas[:,i] = (pca_wt_params[i]**weight_exponent)
    comp_lambdas = comp_lambdas.T.flatten()
    lambdas_IR = np.diag(comp_lambdas)
    
    curve_point_matrix = curve_points.T.flatten()
    
    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    
    if omega is None:
        omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1))
    constraints = [x>=0,cp.sum(x)<=omega]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def optimize_maps_pca_gurobi(pcdf,w0,pcavals_broad,pca_rate_exceedance, pca_wt_params,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_comp_indices=None,
                     weight_exponent = -1.0,
                     comp_weights = None,
                     k = 20,
                     verbose=True,
                     solver = 'GUROBI',
                     MIPGap=0.2,
                     MIPFocus=3,
                     TimeLimit=300):
    
    pcdf = copy.deepcopy(pcdf)
    ncomps = pcdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    comp_curves_mins = np.maximum(pca_rate_exceedance.min(axis=0),10**hz_curves_min)
    comp_curves_maxs = np.minimum(pca_rate_exceedance.max(axis=0),10**hz_curves_max)
    
    curve_points = np.array([np.geomspace(comp_curves_mins[compi], comp_curves_maxs[compi],n_curve_points) for compi in range(ncomps)]).T
    
    pca_prob_exceedance = 1-poisson.pmf(k=0,mu=pca_rate_exceedance)
    pca_prob_exceedance = pca_prob_exceedance/pca_prob_exceedance.max()
    symmetric_quants = np.linspace(0.005,1-0.005,n_curve_points)
    im_symmetric_quants = np.array([interp1d(pca_prob_exceedance[:,i],pcavals_broad[:,i])(symmetric_quants) for i in range(ncomps)]).T
    comp_curve_points = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(im_symmetric_quants[:,i]) for i in range(ncomps)]).T
    curve_points[:,1:] = comp_curve_points[:,1:]
    
    ims = np.array([interp1d(pca_rate_exceedance[:,i],pcavals_broad[:,i])(curve_points[:,i]) for i in range(ncomps)])
    
    if candies_per_event is None:
        imcandy = pcdf
    else:
        imcandy = pcdf.loc[:,:,:candies_per_event-1]
        
    if not candies_comp_indices is None:
        imcandy = imcandy.iloc[:,candies_comp_indices]
        ims = ims[candies_comp_indices,:]
        ncomps = candies_comp_indices.shape[0]

    imcandy_ar = imcandy.values
    
    theta = np.zeros((ncomps*curve_points.shape[0],imcandy.shape[0]))
    for i in range(ncomps):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    # base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    comp_lambdas = curve_points**weight_exponent
    for i in range(1,ncomps):
        comp_lambdas[:,i] = (pca_wt_params[i]**weight_exponent)
    comp_lambdas = comp_lambdas.T.flatten()
    lambdas_IR = np.diag(comp_lambdas)
    
    curve_point_matrix = curve_points.T.flatten()
    
    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    z = cp.Variable(N, boolean = True)
    # if omega is None:
    #     omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1))
    constraints = [x>=0,cp.sum(z)<=k,x<=z]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose,MIPGap=MIPGap,MIPFocus=MIPFocus,TimeLimit=TimeLimit)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def optimize_maps_pca_glpk(pcdf,w0,pcavals_broad,pca_rate_exceedance, pca_wt_params,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_comp_indices=None,
                     weight_exponent = -1.0,
                     comp_weights = None,
                     k = 20,
                     verbose=True,
                     solver = 'GLPK_MI'):
    
    pcdf = copy.deepcopy(pcdf)
    ncomps = pcdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    comp_curves_mins = np.maximum(pca_rate_exceedance.min(axis=0),10**hz_curves_min)
    comp_curves_maxs = np.minimum(pca_rate_exceedance.max(axis=0),10**hz_curves_max)
    
    curve_points = np.array([np.geomspace(comp_curves_mins[compi], comp_curves_maxs[compi],n_curve_points) for compi in range(ncomps)]).T
    
    pca_prob_exceedance = 1-poisson.pmf(k=0,mu=pca_rate_exceedance)
    pca_prob_exceedance = pca_prob_exceedance/pca_prob_exceedance.max()
    symmetric_quants = np.linspace(0.005,1-0.005,n_curve_points)
    im_symmetric_quants = np.array([interp1d(pca_prob_exceedance[:,i],pcavals_broad[:,i])(symmetric_quants) for i in range(ncomps)]).T
    comp_curve_points = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(im_symmetric_quants[:,i]) for i in range(ncomps)]).T
    curve_points[:,1:] = comp_curve_points[:,1:]
    
    ims = np.array([interp1d(pca_rate_exceedance[:,i],pcavals_broad[:,i])(curve_points[:,i]) for i in range(ncomps)])
    
    if candies_per_event is None:
        imcandy = pcdf
    else:
        imcandy = pcdf.loc[:,:,:candies_per_event-1]
        
    if not candies_comp_indices is None:
        imcandy = imcandy.iloc[:,candies_comp_indices]
        ims = ims[candies_comp_indices,:]
        ncomps = candies_comp_indices.shape[0]

    imcandy_ar = imcandy.values
    
    theta = np.zeros((ncomps*curve_points.shape[0],imcandy.shape[0]))
    for i in range(ncomps):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    # base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    comp_lambdas = curve_points**weight_exponent
    for i in range(1,ncomps):
        comp_lambdas[:,i] = (pca_wt_params[i]**weight_exponent)
    comp_lambdas = comp_lambdas.T.flatten()
    lambdas_IR = np.diag(comp_lambdas)
    
    curve_point_matrix = curve_points.T.flatten()
    
    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    z = cp.Variable(N, boolean = True)
    # if omega is None:
    #     omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1))
    constraints = [x>=0,cp.sum(z)<=k,x<=z]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose,tmlim=100)
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def optimize_maps_pca_scip(pcdf,w0,pcavals_broad,pca_rate_exceedance, pca_wt_params,
                     hz_curve_log10_range = [-5,-0.5],
                     n_curve_points = 50,
                     candies_per_event = None,
                     candies_comp_indices=None,
                     weight_exponent = -1.0,
                     comp_weights = None,
                     k = 20,
                     verbose=True,
                     solver = 'SCIP'):
    
    pcdf = copy.deepcopy(pcdf)
    ncomps = pcdf.shape[1]
    
    hz_curves_min,hz_curves_max = hz_curve_log10_range
    comp_curves_mins = np.maximum(pca_rate_exceedance.min(axis=0),10**hz_curves_min)
    comp_curves_maxs = np.minimum(pca_rate_exceedance.max(axis=0),10**hz_curves_max)
    
    curve_points = np.array([np.geomspace(comp_curves_mins[compi], comp_curves_maxs[compi],n_curve_points) for compi in range(ncomps)]).T
    
    pca_prob_exceedance = 1-poisson.pmf(k=0,mu=pca_rate_exceedance)
    pca_prob_exceedance = pca_prob_exceedance/pca_prob_exceedance.max()
    symmetric_quants = np.linspace(0.005,1-0.005,n_curve_points)
    im_symmetric_quants = np.array([interp1d(pca_prob_exceedance[:,i],pcavals_broad[:,i])(symmetric_quants) for i in range(ncomps)]).T
    comp_curve_points = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(im_symmetric_quants[:,i]) for i in range(ncomps)]).T
    curve_points[:,1:] = comp_curve_points[:,1:]
    
    ims = np.array([interp1d(pca_rate_exceedance[:,i],pcavals_broad[:,i])(curve_points[:,i]) for i in range(ncomps)])
    
    if candies_per_event is None:
        imcandy = pcdf
    else:
        imcandy = pcdf.loc[:,:,:candies_per_event-1]
        
    if not candies_comp_indices is None:
        imcandy = imcandy.iloc[:,candies_comp_indices]
        ims = ims[candies_comp_indices,:]
        ncomps = candies_comp_indices.shape[0]

    imcandy_ar = imcandy.values
    
    theta = np.zeros((ncomps*curve_points.shape[0],imcandy.shape[0]))
    for i in range(ncomps):
        theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1
    
    # base_curve_points = np.geomspace(10**hz_curves_min, 10**hz_curves_max,n_curve_points)
    comp_lambdas = curve_points**weight_exponent
    for i in range(1,ncomps):
        comp_lambdas[:,i] = (pca_wt_params[i]**weight_exponent)
    comp_lambdas = comp_lambdas.T.flatten()
    lambdas_IR = np.diag(comp_lambdas)
    
    curve_point_matrix = curve_points.T.flatten()
    
    N=imcandy.shape[0]

    U = theta
    x = cp.Variable(N)
    z = cp.Variable(N, boolean = True)
    # if omega is None:
    #     omega = w0.sum()
    objective = cp.Minimize(cp.norm(lambdas_IR@(curve_point_matrix - U@x),1))
    constraints = [x>=0,cp.sum(z)<=k,x<=z]
    prob = cp.Problem(objective, constraints)
    x.value = w0
    # optimal_value = prob.solve(solver ='SCS',eps=1e-8,eps_infeas=1e-9,verbose=verbose,warm_start=True)
    optimal_value = prob.solve(solver = solver,verbose=verbose,scip_params = {"limits/time" : 300})
    # print("t=",t.value)
    # print("x=",x.value)
    print("val=",optimal_value)
    
    return {'intensity_df':imcandy, 
            'original_weight':w0,
            'optimal_weight':x.value,
            'interpolated_ims':ims,
            'interpolated_return_periods':curve_points}

def get_rhos(imdf,w):
    sites = imdf.columns
    pairs = np.array(list(combinations(sites,2)))
    rhopairs = np.zeros(pairs.shape[0])
    for ip,pair in enumerate(pairs):
        numerator1 = (imdf.loc[:,pair[0]]*imdf.loc[:,pair[1]]*w).sum()
        numerator2 = ((imdf.loc[:,pair[0]]*w).sum())*((imdf.loc[:,pair[1]]*w).sum())
        denomenator1 = ( (((imdf.loc[:,pair[0]]**2)*w).sum()) - (((imdf.loc[:,pair[0]])*w).sum())**2 )**0.5
        denomenator2 = ( (((imdf.loc[:,pair[1]]**2)*w).sum()) - (((imdf.loc[:,pair[1]])*w).sum())**2 )**0.5
        # denomenator1 = (np.maximum(0, (((imdf.loc[:,pair[0]]**2)*w).sum()) - (((imdf.loc[:,pair[0]])*w).sum())**2 ))**0.5
        # denomenator2 = (np.maximum(0, (((imdf.loc[:,pair[1]]**2)*w).sum()) - (((imdf.loc[:,pair[1]])*w).sum())**2 ))**0.5
        if np.isnan(denomenator2):
            print(pair,w.sum())
        rhopairs[ip] = (numerator1-numerator2)/(denomenator1*denomenator2)
    return (pairs,rhopairs)

def get_MSCE(imdf,w0,w2):
    pairs0,rhopairs0 = get_rhos(imdf,w0)
    pairs1,rhopairs1 = get_rhos(imdf,w2)
    SCE = np.abs(rhopairs0-rhopairs1)
    return SCE.mean(),SCE.std() 

def get_MSCE1(imdf,w0,w2,q = [0.05,0.1,0.5,0.9,0.95]):
    ab0 = weightstats.DescrStatsW(imdf,weights=w0)
    corrM0 = ab0.corrcoef
    nzind = np.nonzero(w2)[0]
    ab1 = weightstats.DescrStatsW(imdf.values[nzind,:],weights=w2[nzind])
    corrM1 = ab1.corrcoef
    SCE = np.abs(corrM0-corrM1)/2
    return np.nanmean(SCE),np.nanquantile(SCE,q) 

def get_hzerror(hzvals,imvals,hzbase,hztest,q = [0.05,0.1,0.5,0.9,0.95]):
    nsites = imvals.shape[1]
    yreal = np.array([interp1d(np.concatenate([[1.0],hzbase[:,i]]),np.concatenate([[1e-5],imvals[:,i]]))(hzvals) for i in range(nsites)]).T
    yest = np.array([interp1d(np.concatenate([[1.0],hztest[:,i]]),np.concatenate([[1e-5],imvals[:,i]]))(hzvals) for i in range(nsites)]).T
    errors = np.abs((yreal - yest)/yest)
    emean = errors.mean()
    equants = np.quantile(errors.mean(0),q)
    return emean,equants
    
def get_losserror(loss_hzvals,lossvals,loss_hzbase,loss_hztest,q = [0.05,0.1,0.5,0.9,0.95]):
    nproxies = lossvals.shape[1]
    yreal = np.array([interp1d(np.concatenate([[1.0],loss_hzbase[:,i]]),np.concatenate([[1e-5],lossvals[:,i]]))(loss_hzvals) for i in range(nproxies)]).T
    yest = np.array([interp1d(np.concatenate([[1.0],loss_hztest[:,i]]),np.concatenate([[1e-5],lossvals[:,i]]))(loss_hzvals) for i in range(nproxies)]).T
    errors = np.abs((yreal - yest)/yest)
    emean = errors.mean()
    equants = np.quantile(errors.mean(0),q)
    return emean,equants                
    