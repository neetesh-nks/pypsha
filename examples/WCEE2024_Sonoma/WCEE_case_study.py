# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:21:41 2023

@author: Neetesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from itertools import combinations
from scipy.stats import poisson
from pypsha import psha,utils
import time
import eq_utils as equ
import matplotlib.cm as cm

from wpca import WPCA
from scipy.interpolate import interp1d
from scipy.stats import gmean

#%% Create site

minlat = 38.37
minlong = -123.20
maxlat = 38.71
maxlong = -122.75

m2ddlat = 111111
m2ddlong = np.cos(np.mean((minlat,maxlat))*np.pi/180)*111111
cell = 2000 #meter
reslat = cell/m2ddlat
reslong = cell/m2ddlong

wcee_site = psha.PSHASite(name = 'wcee_grid',
                           grid_specs = [minlong,minlat,maxlong,maxlat,reslong,reslat],
                           erf=1,
                           intensity_measures=[1,3],
                           spectral_periods=[1.0],
                           attenuations = [3],
                           overwrite=True)

# %% Run OpenSHA

wcee_site.write_opensha_input(overwrite=True)
wcee_site.run_opensha(write_output_tofile = True,
                      overwrite=True)

wcee_site.pickle()
wcee_site = psha.PSHASite.unpickle('master/wcee_grid.pickle')

# %% Generate samples and hazard curves

np.random.seed(123)
event_set = psha.PshaEventSet(wcee_site)
nmaps = 10 #per erf row
sa_intensity_ids = ["CB2014_PGA","CB2014_SA_1.0"]
sa_id = sa_intensity_ids[0]
event_set.generate_sa_maps(sa_intensity_ids, nmaps)
im_vals = np.logspace(-3,1,100) #pga range
event_set.generate_hazard_curves(sa_intensity_ids,im_vals)
utils.pickle_obj(event_set)

# event_set = utils.unpickle('event_set_wcee_grid.pickle')
imvals_broad, rate_exceedance = event_set.hazard_curves[sa_id]
test_sites = np.arange(4)
# analytical_prob_exeedance = np.array([utils.probability_of_exceedance(sa, event_set, 'CB2014_PGA', 'site0') for sa in im_vals])

#%% plot hz

fig, ax_lst = plt.subplots(1, 1)
fig.set_figwidth(6)
fig.set_figheight(6)
ax_lst.set_yscale("log")
ax_lst.set_xscale("log")
ax_lst.set_xlabel("PGA")
ax_lst.set_ylabel("Annual rate of exceedance")
ax_lst.set_ylim([10**-4, 10**-1])
ax_lst.set_xlim([im_vals.min(), im_vals.max()])
ax_lst.grid()
colors = cm.rainbow(np.linspace(0, 1, len(test_sites)))
for m,c in zip(np.arange(test_sites.shape[0]),colors):
    ax_lst.plot(imvals_broad[:,m],rate_exceedance[:,m],label = "test_site" + str(int(m)),color=c)
    # ax_lst.plot(imvals_broad[:,i],rate_exceedance1[:,i])
# ax_lst.legend()
#%% Optimization

t0 = time.time()         

nsites = wcee_site.output.sites.shape[0]
xinline = np.where(np.diff(np.diff(wcee_site.output.sites.x))>1e-5)[0][0] + 1
extent_ratios = np.array([0.05,0.1])
nsite_candies = np.zeros_like(extent_ratios)
ks = np.array([16,64,256,512,1028])
MSCE = np.zeros((extent_ratios.shape[0],ks.shape[0]))
SSCE = np.zeros((extent_ratios.shape[0],ks.shape[0]))

w1s = np.zeros((extent_ratios.shape[0],event_set.events.metadata.shape[0]))
w2s = np.zeros((extent_ratios.shape[0],ks.shape[0],event_set.events.metadata.shape[0]))

for ires,res in enumerate(extent_ratios):
    if np.abs(res - 1.0) < 1e-5:
        site_ind = None
        nsite_candies[ires] = nsites
    else:
        site_ind = (np.tile(np.arange(0,int(xinline*res))*xinline,(int(xinline*res),1)).T +
                np.tile(np.arange(0,int(xinline*res)),(int(xinline*res),1))).flatten()
        nsite_candies[ires] =  site_ind.shape[0]
        print(nsite_candies)

    event_set.optimize_sa_maps(sa_id,candies_per_event = 1,
                               candies_site_indices=site_ind,
                               log10_hz_curve_res = 0.25,
                               hz_curve_log10_range = [-3.5,-1.5],
                               verbose = False)
    
    imdf = event_set.optimization_output.intensity_df
    w0 = event_set.optimization_output.original_weight
    w1 = event_set.optimization_output.optimal_weight
    w1s[ires,:] = w1
    for ik,k in enumerate(ks):
        w2 = utils.clean_renormalize_weights(w0,w1,k=k)
        msce,ssce = equ.get_MSCE1(imdf, w0, w2)
        print('NcandySites',nsite_candies[ires],'Nmaps',k,'Error',msce,ssce)
        MSCE[ires,ik]=msce
        SSCE[ires,ik]=ssce
        w2s[ires,ik,:] = w2
        
t1 = time.time()    
    # im1,hz1 = utils.hazard_curves_from_weights(imdf,w2,event_set.events.metadata,imvals = np.logspace(-3,1,100))

#%% Plot
fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_yscale("log")
ax_lst.set_xscale("log")
ax_lst.set_xlabel("PGA")
ax_lst.set_ylabel("Probability of exceedance")
ax_lst.set_ylim([10**-8, 10**0])
ax_lst.set_xlim([10**-3, 10**1])
ax_lst.grid()
#ax_lst.plot(im_vals,analytical_prob_exeedance,'b-',linewidth=2, label='Analytical')
ax_lst.plot(imvals_broad[:,0],1-poisson.pmf(k=0,mu=rate_exceedance[:,0]),'k--',linewidth=1.5,label='Sampled')
# ax_lst.plot(im1[:,0],hz1[:,0],'r',label='Optimized')
ax_lst.legend()

fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_xlabel("Number of sites")
ax_lst.set_ylabel("MSCE")
ax_lst.grid()
ax_lst.plot(nsite_candies,MSCE[:,0],linewidth=2, label='k=16')
ax_lst.plot(nsite_candies,MSCE[:,1],linewidth=2, label='k=64')
ax_lst.plot(nsite_candies,MSCE[:,2],linewidth=2, label='k=256')
ax_lst.plot(nsite_candies,MSCE[:,3],linewidth=2, label='k=512')
ax_lst.legend()
fig.savefig("MSCEvsSites.png")

fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_xlabel("Number of maps")
ax_lst.set_ylabel("MSCE")
ax_lst.grid()
ax_lst.plot(ks,MSCE[0,:],linewidth=2, label='nsites=4')
ax_lst.plot(ks,MSCE[1,:],linewidth=2, label='nsites=9')
ax_lst.plot(ks,MSCE[2,:],linewidth=2, label='nsites=16')
ax_lst.plot(ks,MSCE[3,:],linewidth=2, label='nsites=16')
ax_lst.legend()
fig.savefig("MSCEvsMaps.png")

#%% PCA

imdf = event_set.maps[sa_id]
arr = np.log(imdf.values)
# arr = StandardScaler().fit_transform(arr,sample_weight=sample_weights)
n_comps=8
weights = event_set.events.metadata.annualized_rate.values
sample_weights = np.repeat(weights, nmaps)/nmaps
sample_weights = sample_weights/sample_weights.sum()
sample_weights_broad = np.tile(sample_weights.reshape((-1,1)), reps=arr.shape[1])
pca = WPCA(n_components=n_comps)
principalcomponents = pca.fit_transform(arr,weights=sample_weights_broad)

#%%
topimrow = np.zeros((1,imvals_broad.shape[1]))
tophzrow = rate_exceedance[0:1,:] + 1e-10
imb = np.concatenate((topimrow, imvals_broad), axis=0)
re = np.concatenate((tophzrow, rate_exceedance), axis=0)
lamb_weights = np.array([interp1d(imb[:,i],re[:,i])(imdf.values[:,i]) for i in range(nsites)])
gmean_lamb_weights = gmean(lamb_weights,axis=0)

#%% pca hz
cols = np.char.add('p',np.arange(1,n_comps+1).astype(str))
principaldf = pd.DataFrame(data = -1*principalcomponents
             , columns = cols, index=imdf.index)
pcavals = np.linspace(principaldf.min().min(),principaldf.max().max(),1024)
pcavals_broad,pca_rate_exceedance = utils.hazard_curves_from_weights(principaldf,weights,event_set.events.metadata,pcavals)

pcalamb = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(principaldf.values[:,i]) for i in range(n_comps)])

pca_prob_exceedance = 1-poisson.pmf(k=0,mu=pca_rate_exceedance)
pca_prob_exceedance = pca_prob_exceedance/pca_prob_exceedance.max()
symmetric_quants = np.linspace(0.001,1-0.001,20)
im_symmetric_quants = np.array([interp1d(pca_prob_exceedance[:,i],pcavals_broad[:,i])(symmetric_quants) for i in range(n_comps)]).T
comp_curve_points = np.array([interp1d(pcavals_broad[:,i],pca_rate_exceedance[:,i])(im_symmetric_quants[:,i]) for i in range(n_comps)]).T

# plt.plot(pcavals_broad[:,1],-np.diff(pca_prob_exceedance[:,1], append=0))
# plt.plot(pcavals_broad[:,1],-np.diff(pca_rate_exceedance[:,1], append=0)/(-(np.diff(pca_rate_exceedance[:,1], append=0).min())))
# plt.plot(pcavals_broad[:,1],pca_prob_exceedance[:,1])

#%%

xseq = np.geomspace(0.1, 1e-6, num=60)
pca_fit_params = np.array([np.polyfit(pcalamb[i,:],gmean_lamb_weights, deg=1) for i in range(n_comps)])
fig, ax_lsts = plt.subplots(3, 2)
fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
fig.set_figheight(15)
fig.set_figwidth(10)
for l in range(3):
    for j in range(2):
        ax_lst = ax_lsts[l,j]
        ax_lst.set_yscale("log")
        ax_lst.set_xscale("log")
        i = l*2 + j
        ax_lst.set_xlabel("Reccurrence rate PC " + str(i))
        ax_lst.grid()
        ax_lst.set_ylabel("Marginal rate geometric mean")
        ax_lst.set_ylim([10**-4, 10**-1])
        ax_lst.set_xlim([10**-4, 10**-1])
        try:
            ax_lst.plot(pcalamb[i,:],gmean_lamb_weights,'.')
            ax_lst.plot(xseq, pca_fit_params[i,-1] + pca_fit_params[i,-2] * xseq, "k", lw=2.5, label =  "Linear fit")
        except:
            continue
        ax_lst.legend()
fig.savefig("PrincipalComponentHAzardCurve_vs_original.png", dpi = 600)
#%% opt

omegas = np.geomspace(0.001, 0.05,5)
ks_pca = np.zeros_like(omegas)
w1s_pca = np.zeros((weights.shape[0],omegas.shape[0]))
w2s_pca = np.zeros_like(w1s_pca)
for oid, omega in enumerate(omegas):
    # omega = 0.0023
    candies_per_event = 1
    n_curve_points = 100
    hz_curve_log10_range = [-4,-1]
    
    result = equ.optimize_maps_pca(principaldf, weights, pcavals_broad, pca_rate_exceedance,pca_fit_params[:,1],
                                        hz_curve_log10_range = hz_curve_log10_range,
                                        n_curve_points = n_curve_points,
                                        candies_per_event = candies_per_event,
                                        omega = omega,
                                        verbose = False)
    
    
    w0 = result['original_weight']
    w1 = result['optimal_weight']
    pccandy = result['intensity_df']
    
    
    k = (w1>1e-8).sum()
    print(k)
    ks_pca[oid] = k
    nonzero_ids = np.where(w1>1e-8)[0]
    principaldf_sub = pd.DataFrame(pccandy.values[nonzero_ids,:],columns=pccandy.columns)
    w0_sub = np.tile(weights,candies_per_event)[nonzero_ids]
    
    # print(oid,round(omega,5),k)
    result1 = equ.optimize_maps(principaldf_sub, w0_sub, pcavals_broad, pca_rate_exceedance,       
                                    hz_curve_log10_range = hz_curve_log10_range,
                                    n_curve_points = n_curve_points,
                                    omega = weights.sum(),
                                    verbose = False)
    w1_sub = result1['optimal_weight']
    w1_sub[w1_sub<0.0] = 0.0
    w1_new = np.zeros_like(np.tile(weights,candies_per_event))
    w1_new[nonzero_ids] = w1_sub
    w1s_pca[:,oid] = w1_new
    w2s_pca[:,oid] = w1_new*(weights.sum()/w1_new.sum())

#%%
fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_xlabel("Omega")
ax_lst.set_ylabel("Number of scenarios")
ax_lst.grid()
ax_lst.plot(omegas,ks_pca,linewidth=2)
ax_lst.legend()
fig.savefig("OmegaVsNscenarios.png")

#%%
for oid, omega in enumerate(omegas[:]):
    im1,hz1 = utils.hazard_curves_from_weights(pccandy,w2s_pca[:,oid],event_set.events.metadata,pcavals)

    fig, ax_lsts = plt.subplots(3, 2)
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    fig.set_figheight(15)
    fig.set_figwidth(10)
    for l in range(3):
        for j in range(2):
            ax_lst = ax_lsts[l,j]
            ax_lst.set_yscale("log")
            # ax_lst.set_xscale("log")
    #        ax_lst.set_xlabel("PGA")
    #        ax_lst.set_ylabel("Probability of exceedance")
            ax_lst.set_ylim([10**-4, 10**-0.5])
            ax_lst.set_xlim([principaldf.min().min(), principaldf.max().max()])
            ax_lst.grid()
            i = l*2 + j
            ax_lst.plot(pcavals_broad[:,i],pca_rate_exceedance[:,i],linewidth=1.5,label='Fullset')
            ax_lst.plot(im1[:,i],hz1[:,i],'--',label='Subset')
            ax_lst.set_title("pc_" + str(i+1))
            # ax_lst.plot(im1[:,i],hz2[:,i],label='Subset scaled')
    fig.suptitle("Tuning parameter omega=" + str(round(omega,3)) + "  Number of maps k=" + str(int(ks_pca[oid])))
    ax_lst.legend()
    # fig.savefig("Results/EQ/eq_OptimizedScenarios_mine1"+str(ks_pca[oid])+".png", dpi = 600)

#%% corr plot
import eq_utils as equ
MSCE_pca = np.zeros(omegas.shape[0])
SSCE_pca = np.zeros(omegas.shape[0])
imcandy = imdf.loc[:,:,:candies_per_event-1]

for oid, omega in enumerate(omegas[:]):
    imvals_broad_test = imvals_broad[:,test_sites]
    rate_exceedance_test = rate_exceedance[:,test_sites]
    im1, hz1 = utils.hazard_curves_from_weights(imcandy.iloc[:,test_sites],w2s_pca[:,oid],event_set.events.metadata,imvals_broad_test[:,0])
    
    msce,ssce = equ.get_MSCE1(imcandy, weights/weights.sum(), w2s_pca[:,oid])
    MSCE_pca[oid] = msce
    SSCE_pca[oid] = ssce

    fig, ax_lsts = plt.subplots(8, 4)
    fig.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for l in range(8):
        for j in range(4):
            ax_lst = ax_lsts[l,j]
            ax_lst.set_yscale("log")
            ax_lst.set_xscale("log")
            # ax_lst.set_xlabel("PGA")
            # ax_lst.set_ylabel("Probability of exceedance")
            ax_lst.set_ylim([10**-4, 10**-1])
            ax_lst.set_xlim([im_vals.min(), im_vals.max()])
            ax_lst.grid()
            i = l*4 + j
            ax_lst.plot(imvals_broad[:,i],rate_exceedance[:,i],linewidth=1.5,label='Fullset')
            ax_lst.plot(im1[:,i],hz1[:,i],'--',label='Subset')
            ax_lst.set_title("test_site_" + str(i+1))
            # ax_lst.plot(im1[:,i],hz2[:,i],label='Subset scaled')
    fig.suptitle("Tuning parameter omega=" + str(round(omega,3)) + "  Number of maps k=" + str(int(ks_pca[oid])) + "MSCE = " + str(round(msce,5)))
    ax_lst.legend()
    # fig.savefig("Results/EQ/eq_Test_Sites_OptimizedScenarios_mine1"+str(ks_pca[oid])+".png", dpi = 600)

fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_xlabel("Number of scenario maps")
ax_lst.set_ylabel("MSCE")
ax_lst.grid()
ax_lst.fill_between(ks_pca,MSCE_pca-SSCE_pca,MSCE_pca+SSCE_pca,label='+- 1 stdev',facecolor='gray', alpha=0.5)
ax_lst.plot(ks_pca,MSCE_pca,label='MSCE_pca')
ax_lst.legend()
# fig.savefig("Results/EQ/eq_MSCE_mine.png"+str(ks_pca[oid])+".png", dpi = 600)