# -*- coding: utf-8 -*-
"""
Jan 2023

@author: neetesh
"""

import numpy as np
import copy
from scipy.stats import lognorm, poisson
from statsmodels.distributions.empirical_distribution import ECDF

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
        