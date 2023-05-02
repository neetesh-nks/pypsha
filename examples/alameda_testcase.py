# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 13:21:41 2023

@author: Neetesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import poisson
from pypsha import psha,utils

#%% Create site
test_site = psha.PSHASite(name = 'alameda',
                            site_filename = "Alameda_sparce.csv",
                            erf=1,
                            intensity_measures = [1],
                            attenuations = [3,4],
                            overwrite=True)

# %% Run OpenSHA
test_site.write_opensha_input(overwrite = True)
test_site.run_opensha(overwrite= True,
                      write_output_tofile = True)

# test_site.pickle()
# test_site  = psha.PSHASite.unpickle('master/alameda.pickle')

# %% Generate samples and hazard curves

np.random.seed(123)
event_set = psha.PshaEventSet(test_site)
nmaps = 50 #per erf row
sa_intensity_ids = ["CB2014_PGA","CY2014_PGA"]

event_set.generate_sa_maps(sa_intensity_ids, nmaps)
im_vals = np.logspace(-3,1,100) #pga range
event_set.generate_hazard_curves(sa_intensity_ids,im_vals)

sa_id = sa_intensity_ids[0]
SAs, rate_exeedance = event_set.hazard_curves[sa_id]
analytical_prob_exeedance = np.array([utils.probability_of_exceedance(sa, event_set, 'CB2014_PGA', 'site0') for sa in im_vals])


#%% Optimization

event_set.optimize_sa_maps(sa_id,candies_per_event = 1,
                           hz_curve_log10_range = [-4,-1.3],
                           solver='SCS',verbose=True,
                           eps=1e-6,eps_infeas=1e-8,
                           max_iters= 500000)

imdf = event_set.optimization_output.intensity_df
w0 = event_set.optimization_output.original_weight
w1 = event_set.optimization_output.optimal_weight

w2 = utils.clean_renormalize_weights(w0,w1,k=None)
im1,hz1 = utils.hazard_curves_from_weights(imdf,w2,event_set.events.metadata,imvals = np.logspace(-3,1,100))

#%% Plot
fig, ax_lst = plt.subplots(1, 1)
ax_lst.set_yscale("log")
ax_lst.set_xscale("log")
ax_lst.set_xlabel("PGA")
ax_lst.set_ylabel("Probability of exceedance")
ax_lst.set_ylim([10**-8, 10**0])
ax_lst.set_xlim([10**-3, 10**1])
ax_lst.grid()
ax_lst.plot(im_vals,analytical_prob_exeedance,'b-',linewidth=2, label='Analytical')
ax_lst.plot(SAs[:,0],1-poisson.pmf(k=0,mu=rate_exeedance[:,0]),'k--',linewidth=1.5,label='Sampled')
ax_lst.plot(im1[:,0],hz1[:,0],'r',label='Optimized')
ax_lst.legend()
