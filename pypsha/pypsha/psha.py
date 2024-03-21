# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:26:09 2023

@author: neetesh
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import os
import subprocess
import shutil
import re
import pickle
import copy

from itertools import combinations, combinations_with_replacement
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial.distance import squareform
from scipy.stats import multivariate_normal, norm
from statsmodels.distributions.empirical_distribution import ECDF
from typing import Optional

from pypsha.utils import haversine_distance
from pypsha.find_modules import _find_java_module

class SiteFiles:
    def __init__(self):
        self.user = None
        self.master = None
        self.opensha = None
        self.site_filename = None
        self.opensha_inputfile = None
        self.opensha_output = None

class SiteGroundMotionParams:
    def __init__(self):
        self.erf = None
        self.background_seismicity = None
        self.attenuations = None
        self.rupture_offset = None
        self.intensity_measures = None
        self.spectral_periods = None

class SiteOutput:
    def __init__(self):
        self.sites = None
        self.metadata = None
        self.intensity_filelist = None
        self.data = None

class PSHASite:

    ERF_OPTIONS = {
        1: 'WGCEP (2007) UCERF2 - Single Branch',
        2: 'USGS/CGS 2002 Adj. Cal. ERF',
        3: 'WGCEP UCERF 1.0 (2005)',
        4: 'GEM1 CEUS ERF'
    }
    
    BACKGROUND_SEISMICITY_OPTIONS = {
        1: 'Include',
        2: 'Exclude',
        3: 'Only Background'
    }
    
    ATTENUATIONS_OPTIONS = {
        1: "Abrahamson, Silva & Kamai (2014)",
        2: "Boore, Stewart, Seyhan & Atkinson (2014)",
        3: "Campbell & Bozorgnia (2014)",
        4: "Chiou & Youngs (2014)",
        5: "Idriss (2014)",
        6: 'Campbell & Bozorgnia (2008)',
        7: 'Boore & Atkinson (2008)',
        8: 'Abrahamson & Silva (2008)',
        9: 'Chiou & Youngs (2008)',
        10: 'Boore & Atkinson (2006)',
        11: 'Chiou & Youngs (2006)',
        12: 'Campbell & Bozorgnia (2006)',
        13: 'Boore, Joyner & Fumal (1997)',
        14: 'Field (2000)',
        15: 'ShakeMap (2003)'
    }
    
    IMS_OPTIONS = {
        1: 'PGA',
        2: 'PGV',
        3: 'SA'
    }
    
    def __init__(self, name, path=None, site_filename=None, grid_specs=None,
                 erf=1, background_seismicity=2, attenuations=[3],
                 rupture_offset=5, intensity_measures=[1], spectral_periods=[0.3],
                 overwrite=False, **kwargs):
        
        self.name = name
        
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
        
        self.files = SiteFiles()
        self.files.user = os.path.join(self.path, 'user')
        self.files.master = os.path.join(self.path, 'master')
        self.files.opensha = os.path.join(self.path, 'opensha')
        self.create_path(self.files.user)
        self.create_path(self.files.master)
        self.create_path(self.files.opensha)
        
        if site_filename is None:
            if grid_specs is None:
                raise ValueError("Provide site_filename or grid_specs as [minx, miny, maxx, maxy, resx, resy]")
            else:
                site_filename = self.generate_grid(grid_specs, overwrite, **kwargs)
        
        self.files.site_filename = site_filename
        site_filepath = os.path.join(self.files.user, site_filename)
        if not os.path.exists(site_filepath):
            shutil.copy(site_filename, site_filepath)
        
        self.ground_motion_params = SiteGroundMotionParams()
        self.ground_motion_params.erf = erf
        self.ground_motion_params.background_seismicity = background_seismicity
        self.ground_motion_params.attenuations = attenuations
        self.ground_motion_params.rupture_offset = rupture_offset
        self.ground_motion_params.intensity_measures = intensity_measures
        self.ground_motion_params.spectral_periods = spectral_periods
    
    def create_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def pickle(self, folder_path=None):
        if folder_path is None:
            folder_path = self.files.master
        filename = os.path.join(folder_path, f"{self.name}.pickle")
        with open(filename, "wb") as filehandler:
            pickle.dump(self, filehandler)

    def unpickle(filepath):
        with open(filepath, "rb") as filehandler:
            site_object = pickle.load(filehandler)
        return site_object

    def generate_grid(self, grid_specs, overwrite=False, **kwargs):
        constant_vs30 = kwargs.get('constant_vs30', 760)
        
        minx, miny, maxx, maxy, resx, resy = grid_specs
        
        xvals = np.arange(minx, maxx, resx)
        yvals = np.arange(miny, maxy, resy)
        xgrid, ygrid = np.meshgrid(xvals, yvals)
        xlin = xgrid.reshape(-1, 1)
        ylin = ygrid.reshape(-1, 1)
        vs30 = np.ones_like(xlin) * constant_vs30
        site_df = pd.DataFrame(np.concatenate([xlin, ylin, vs30], axis=1), columns=['x', 'y', 'vs30'])
        
        filename = f"{self.name}_site.csv"
        filepath = os.path.join(self.files.user, filename)
        
        if not os.path.exists(filepath) or overwrite:     
            site_df.to_csv(filepath, index=False)  
        else:
            raise Warning("Site file from grid already exists, provide overwrite=True to overwrite")
        
        return filename

    def write_opensha_input(self, overwrite=False):
        filename = f"{self.name}_opensha_input.txt"
        filepath = os.path.join(self.files.opensha, filename)
        
        if not os.path.exists(filepath):
            self.write_file(filepath)
        elif overwrite:
            os.remove(filepath)
            self.write_file(filepath)
        else:
            raise Warning("OpenSHA input file already exists, provide overwrite=True to overwrite")
        
        self.files.opensha_inputfile = filename
        return filename
    
    def write_file(self, filepath):
        site_df = pd.read_csv(os.path.join(self.files.user,self.files.site_filename))
        self.output = SiteOutput()
        self.output.sites = site_df
        
        erf = self.ground_motion_params.erf
        background_seismicity = self.ground_motion_params.background_seismicity
        rupture_offset = self.ground_motion_params.rupture_offset
        attenuations = np.array(self.ground_motion_params.attenuations)
        intensity_measures = np.array(self.ground_motion_params.intensity_measures)
        spectral_periods = np.array(self.ground_motion_params.spectral_periods)
        
        with open(filepath, 'w+') as file:
            file.write(self.ERF_OPTIONS[erf] + '\n')
            file.write(self.BACKGROUND_SEISMICITY_OPTIONS[background_seismicity] + '\n')
            file.write(str(rupture_offset) + '\n')
            
            file.write(str(attenuations.shape[0]) + '\n')
            for attenuation in attenuations:
                file.write(self.ATTENUATIONS_OPTIONS[attenuation] + '\n')
            
            is_sa = int(3 in intensity_measures)
            ims_length = intensity_measures.shape[0] - is_sa + is_sa*spectral_periods.shape[0]
            file.write(str(ims_length) + '\n')
            
            for im in intensity_measures:
                im_str = self.IMS_OPTIONS[im]
                if im_str=='SA':
                    for period in spectral_periods:
                        file.write(im_str + ' ' + str(period) + '\n')
                else:
                    file.write(im_str + '\n')
            
            file.write(str(site_df.shape[0]) + '\n')
        
        site_df.loc[:,['y','x','vs30']].to_csv(filepath, header=None, index=None, sep=' ', mode='a')

    def run_opensha(self, outdir_name: Optional[str] = None, overwrite: bool = False, jarpath: Optional[str] = None, write_output_tofile: bool = False) -> None:
        if outdir_name is None:
            outdir_name = f"{self.name}_opensha_output"
        self.files.opensha_output = outdir_name
    
        current_dir = os.getcwd()
        os.chdir(self.files.opensha)
    
        if jarpath is None:
            jarpath = _find_java_module()
    
        if (not os.path.exists(outdir_name)) or overwrite:
            subprocess.call(["java", "-jar", "-Xmx500M", jarpath, self.files.opensha_inputfile, outdir_name])
        else:
            raise Warning("OpenSHA output folder already exists, provide overwrite True to overwrite")
        assert os.path.exists(outdir_name)
        assert len(os.listdir(outdir_name)) > 3
    
        temp_df = pd.read_csv(os.path.join(outdir_name, "src_rup_metadata.txt"), header=None, sep="  ", engine="python")
        id1 = temp_df.iloc[:, 0].values.astype(int)
        temp1_df = temp_df.iloc[:, 1].apply(lambda x: pd.Series(x.split(" ")))
        id2 = temp1_df.iloc[:, 0].astype(int).values
        lamb = temp1_df.iloc[:, 1].astype(float).values
        index_df = pd.MultiIndex.from_arrays([id1, id2], names=["source_id", "rupture_id"])
        metadata_df = pd.DataFrame(lamb, index=index_df, columns=["annualized_rate"])
        metadata_df["magnitude"] = temp_df.iloc[:, 2].values.astype(float)
        metadata_df["name"] = temp_df.iloc[:, 3].apply(lambda x: re.sub("[^A-Za-z0-9]+", "_", x)).values
    
        intensity_filelist = os.listdir(outdir_name)
        for file in ["rup_dist_info.txt", "rup_dist_jb_info.txt", "src_rup_metadata.txt"]:
            intensity_filelist.remove(file)
    
        self.output.metadata = metadata_df
    
        nsites = self.output.sites.shape[0]
        im_dfs = []
        for file in intensity_filelist:
            temp_df = pd.read_csv(os.path.join(outdir_name, file), sep=" ", header=None)
            im_df = pd.concat(
                {
                    "mu": pd.DataFrame(
                        temp_df.iloc[:, np.arange(2, temp_df.shape[1], 3)].values,
                        index=index_df,
                        columns=np.char.add("site", np.arange(nsites).astype("str")),
                    ),
                    "sigma": pd.DataFrame(
                        temp_df.iloc[:, np.arange(3, temp_df.shape[1], 3)].values,
                        index=index_df,
                        columns=np.char.add("site", np.arange(nsites).astype("str")),
                    ),
                    "tau": pd.DataFrame(
                        temp_df.iloc[:, np.arange(4, temp_df.shape[1], 3)].values,
                        index=index_df,
                        columns=np.char.add("site", np.arange(nsites).astype("str")),
                    ),
                },
                names=["descriptor"],
            )
            im_dfs.append(im_df)
    
        data_df = pd.concat(im_dfs, keys=[file[:-4] for file in intensity_filelist], names=["im_map"])
        self.output.intensity_filelist = pd.DataFrame(intensity_filelist, columns=["filename"])
        self.output.data = data_df
        os.chdir(current_dir)
        
        if write_output_tofile:
            metadata_path = os.path.join(self.files.master, self.name + '_metadata.csv')
            self.output.metadata.to_csv(metadata_path)
            
            intensity_path = os.path.join(self.files.master, self.name + '_intensity_filelist.csv')
            self.output.intensity_filelist.to_csv(intensity_path, index=False)
            
            data_path = os.path.join(self.files.master, self.name + '_data.csv')
            self.output.data.to_csv(data_path)

class EventSetOutput:
    def __init__(self):
        self.intensity_df = None
        self.original_weight = None
        self.optimal_weight = None
        self.interpolated_ims = None
        self.interpolated_return_periods = None

class PshaEventSet:
    
    def __init__(self, site):
        self.name = f"event_set_{site.name}"
        self.events = site.output
        self.maps = {}
        self.hazard_curves = {}
        self.optimization_output = EventSetOutput()
    
    def get_site_distances(self):
        sites = self.events.sites
        
        indices = np.array(list(combinations(sites.index,2)))
        coords1 = sites.loc[indices[:,0],['x','y']].values
        coords2 = sites.loc[indices[:,1],['x','y']].values
        
        distance_matrix = squareform(haversine_distance(coords1[:,0], coords1[:,1], coords2[:,0], coords2[:,1]))
        return distance_matrix
        
    def get_epsilons(self, period=1.0, maps_size=100):
        
        sites = self.events.sites
        
        indices = np.array(list(combinations(sites.index,2)))
        coords1 = sites.loc[indices[:,0],['x','y']].values
        coords2 = sites.loc[indices[:,1],['x','y']].values
        
        distance_matrix = squareform(haversine_distance(coords1[:,0], coords1[:,1], coords2[:,0], coords2[:,1]))
        
        if period < 1:
            b = 40.7 - 15*period
        elif period >= 1:
            b = 22.0 + 3.7*period
        
        covariance_matrix = np.exp(-3*distance_matrix/ b)
        epsilons = multivariate_normal.rvs(mean=np.zeros(covariance_matrix.shape[0]),cov=covariance_matrix,size=maps_size)
        
        return epsilons

    def baker_jayaram2008 (self,t1,t2,maps_size=100):
         tmin = np.minimum(t1,t2)
         tmax = np.maximum(t1,t2)
         
         c1 = (1-np.cos(np.pi/2 - np.log(tmax/np.maximum(tmin, 0.109)) * 0.366 ))
         c2 = 1 - 0.105*(1 - 1.0/(1+np.exp(100*tmax-5)))*(tmax-tmin)/(tmax-0.0099)
         
         c3 = c1
         c3[tmax<0.109] = c2[tmax<0.109]
         
         c4 = c1 + 0.5 * (c3**0.5 - c3) * (1 + np.cos(np.pi*(tmin)/(0.109)))
         
         rhos = c4
         ind1 = tmax<0.2
         rhos[ind1] = np.minimum(c2,c4)[ind1]
         ind1 = tmin>0.109
         rhos[ind1] = c1[ind1]
         ind1 = tmax<=0.109
         rhos[ind1] = c2[ind1]
         
         covariance_matrix = rhos
         covariance_matrix = np.maximum(covariance_matrix, np.eye(covariance_matrix.shape[0]))
         etas = multivariate_normal.rvs(mean=np.zeros(covariance_matrix.shape[0]),cov=covariance_matrix,size=maps_size)
         
         return etas

    def loth_baker2013 (self,t1,t2,distance,maps_size=100,B_DATA_YEAR=2019):
        
        TIME_PERIODS = np.array([0.01, 0.1, 0.2, 0.5, 1, 2, 5, 7.5, 10.0001])
        
        # Upper triangular flattened values
        B_UNIQUE =   np.array([[0.29, 0.25, 0.23, 0.23, 0.18, 0.1 , 0.06, 0.06, 0.06,
                                0.3 , 0.2 , 0.16, 0.1 , 0.04, 0.03, 0.04, 0.05, 0.27,
                                0.18, 0.1 , 0.03, 0.  , 0.01, 0.02, 0.31, 0.22, 0.14,
                                0.08, 0.07, 0.07, 0.33, 0.24, 0.16, 0.13, 0.12, 0.33,
                                0.26, 0.21, 0.19, 0.37, 0.3 , 0.26, 0.28, 0.24, 0.23],
                               [0.47, 0.4 , 0.43, 0.35, 0.27, 0.15, 0.13, 0.09, 0.12,
                                0.42, 0.37, 0.25, 0.15, 0.03, 0.04, 0.  , 0.03, 0.45,
                                0.36, 0.26, 0.15, 0.09, 0.05, 0.08, 0.42, 0.37, 0.29,
                                0.2 , 0.16, 0.16, 0.48, 0.41, 0.26, 0.21, 0.21, 0.55,
                                0.37, 0.33, 0.32, 0.51, 0.49, 0.49, 0.62, 0.6 , 0.68],
                               [0.24,  0.22,  0.21,  0.09, -0.02,  0.01,  0.03,  0.02,  0.01,
                                0.28,  0.2 ,  0.04, -0.05,  0.  ,  0.01,  0.01, -0.01,  0.28,
                                0.05, -0.06,  0.  ,  0.04,  0.03,  0.01,  0.27,  0.14,  0.05,
                                0.05,  0.05,  0.04,  0.19,  0.07,  0.05,  0.05,  0.05,  0.12,
                                0.08,  0.07,  0.06,  0.12,  0.1 ,  0.08,  0.1 ,  0.09,  0.09]])
    
        B_UNIQUE_OLD =   np.array([[0.3 , 0.24, 0.23, 0.22, 0.16, 0.07, 0.03, 0.  , 0.,
                                    0.27, 0.19,  0.13, 0.08, 0.  , 0.  , 0.  , 0.  , 0.26,
                                    0.19, 0.12, 0.04, 0.  , 0.  , 0.  , 0.32, 0.23, 0.14,
                                    0.09, 0.06, 0.04, 0.32, 0.22, 0.13, 0.09, 0.07, 0.33,
                                    0.23, 0.19, 0.16, 0.34, 0.29, 0.24, 0.3 , 0.25, 0.24],
                                   [0.31,  0.26,  0.27,  0.24,  0.17,  0.11,  0.08,  0.06,  0.05,
                                    0.29,  0.22,  0.15,  0.07,  0.  ,  0.  ,  0.  , -0.03,  0.29,
                                    0.24,  0.15,  0.09,  0.03,  0.02,  0.  ,  0.33,  0.27,  0.23,
                                    0.17,  0.14,  0.14,  0.38,  0.34,  0.23,  0.19,  0.21,  0.44,
                                    0.33,  0.29,  0.32,  0.45,  0.42,  0.42,  0.47,  0.47,  0.54],
                                   [0.38,  0.36,  0.35,  0.17,  0.04,  0.04,  0.  ,  0.03,  0.08,
                                    0.43,  0.35,  0.13,  0.  ,  0.02,  0.  ,  0.02,  0.08,  0.45,
                                    0.11, -0.04, -0.02, -0.04, -0.02,  0.03,  0.35,  0.2 ,  0.06,
                                    0.02,  0.04,  0.02,  0.3 ,  0.14,  0.09,  0.12,  0.04,  0.22,
                                    0.12,  0.13,  0.09,  0.21,  0.17,  0.13,  0.23,  0.1 ,  0.22]])
        
        tt_unique  = list(combinations_with_replacement(TIME_PERIODS,2))
        if B_DATA_YEAR == 2019:
            interp = LinearNDInterpolator(tt_unique, B_UNIQUE.T)
        elif B_DATA_YEAR == 2013:
            interp = LinearNDInterpolator(tt_unique, B_UNIQUE_OLD.T)
    
        tmin = np.minimum(t1,t2)
        tmax = np.maximum(t1,t2)
        
        b_coef = interp(tmin, tmax).T
        covariance_matrix=b_coef[0]*np.exp(-3.0*distance/20.0) + b_coef[1]*np.exp(-3.0*distance/70.0) + b_coef[2]*(distance==0)
        covariance_matrix = np.maximum(covariance_matrix , np.eye(covariance_matrix.shape[0]))
        epsilons = multivariate_normal.rvs(mean=np.zeros(covariance_matrix.shape[0]),cov=covariance_matrix,size=maps_size)
    
        return epsilons

    def generate_maps(self,intensity_id,nmaps):
        
        if intensity_id[-3:] == 'PGA':
            period = 1.0
        elif intensity_id[-3:] == 'PGV':
            period = 0.0
        elif intensity_id.split('_')[-1].replace('.','').isdigit():
            period = float(intensity_id.split('_')[-1])
        else:
            raise Exception("Invalid intensity ID, Check events intensity filename list")

        nsites = self.events.sites.shape[0]
        nevents = self.events.metadata.shape[0]
        
        nmaps1 = nmaps
        if nmaps1 == 1:
            nmaps = 2
        epsilons = self.get_epsilons(period,maps_size=(nevents,nmaps))
        eta_sites = norm.rvs(0,1,size=(nevents,nmaps,1))
        
        medians = np.expand_dims(self.events.data.loc[(intensity_id,'mu'),:].values,axis=1)
        sigmas = np.expand_dims(self.events.data.loc[(intensity_id,'sigma'),:].values,axis=1) #total
        taus = np.expand_dims(self.events.data.loc[(intensity_id,'tau'),:].values,axis=1) #between
        assert taus[0,0,0] > 0.0
        phis = (sigmas**2 - taus**2)**0.5 #within
        
        intensity = np.exp(medians + phis*epsilons + taus*eta_sites)
        
        ids = self.events.metadata.index.to_frame().iloc[:,:].values
        subindex1 = np.repeat(ids, (nmaps),axis=0)
        subindex2 = np.tile(np.arange(nmaps).reshape(-1,1),(ids.shape[0],1))
        idnames = list(self.events.metadata.index.names)
        idnames.append('map_id')
        dfindex = pd.MultiIndex.from_arrays([subindex1[:,0],subindex1[:,1],subindex2.flatten()], names = idnames)
        df = pd.DataFrame(intensity.reshape((-1,nsites)),index=dfindex, columns = np.char.add('site', np.arange(nsites).astype('str')))
        if nmaps1 == 1:
            self.maps[intensity_id] = df.loc[:,:,0,:]
        else:
            self.maps[intensity_id] = df
        return None

    def generate_maps_check(self,intensity_id,nmaps):

        nsites = self.events.sites.shape[0]
        nevents = self.events.metadata.shape[0]
        
        errors = norm.rvs(0,1,size=(nevents,nmaps,1))
        
        medians = np.expand_dims(self.events.data.loc[(intensity_id,'mu'),:].values,axis=1)
        sigmas = np.expand_dims(self.events.data.loc[(intensity_id,'sigma'),:].values,axis=1) #total
        
        intensity = np.exp(medians + sigmas*errors)
        
        ids = self.events.metadata.index.to_frame().iloc[:,:].values
        subindex1 = np.repeat(ids, (nmaps),axis=0)
        subindex2 = np.tile(np.arange(nmaps).reshape(-1,1),(ids.shape[0],1))
        idnames = list(self.events.metadata.index.names)
        idnames.append('map_id')
        dfindex = pd.MultiIndex.from_arrays([subindex1[:,0],subindex1[:,1],subindex2.flatten()], names = idnames)
        df = pd.DataFrame(intensity.reshape((-1,nsites)),index=dfindex, columns = np.char.add('site', np.arange(nsites).astype('str')))
        self.maps[intensity_id] = df
        
        return None

    def generate_sa_maps(self,intensity_ids,nmaps):
        nmaps1 = nmaps
        if nmaps1 == 1:
            nmaps = 2
        periods = []
        intensity_ids_clean = []
        for intensity_id in intensity_ids:        
            if intensity_id[-3:] == 'PGA':
                period = 0.01
            elif intensity_id[-3:] == 'PGV':
                raise Warning("Skipping PGV, Only meant for PGA and Sa")
                continue
            elif intensity_id.split('_')[-1].replace('.','').isdigit():
                period = float(intensity_id.split('_')[-1])
            else:
                raise Exception("Invalid intensity ID, Check events intensity filename list\n Expects Sa id in SA_0.3 like format")
            periods.append(period)
            intensity_ids_clean.append(intensity_id)
        
        periods = np.array(periods)
        nt = periods.shape[0]
        
        site_distance_matrix = self.get_site_distances()
        nsites = site_distance_matrix.shape[0]
        nevents = self.events.metadata.shape[0]
        
        distancearray = np.tile(site_distance_matrix,(nt,nt))
        periods_reps = np.repeat(periods,nsites)
        t1_repsarray,t2_repsarray = np.meshgrid(periods_reps,periods_reps)
        epsilons = self.loth_baker2013(t1_repsarray,t2_repsarray, distancearray,maps_size=(nevents,nmaps))
        
        t1array,t2array = np.meshgrid(periods,periods)
        eta_sites = self.baker_jayaram2008(t1array,t2array,maps_size=(nevents,nmaps))
        
        dfs_dict = {}
        for idi,intensity_id in enumerate(intensity_ids_clean):
            medians = np.expand_dims(self.events.data.loc[(intensity_id,'mu'),:].values,axis=1)
            sigmas = np.expand_dims(self.events.data.loc[(intensity_id,'sigma'),:].values,axis=1)
            taus = np.expand_dims(self.events.data.loc[(intensity_id,'tau'),:].values,axis=1)
            assert taus[0,0,0] > 0.0
            phis = (sigmas**2 - taus**2)**0.5

            intensity = np.exp(medians + phis*epsilons[:,:,idi*nsites:(idi+1)*nsites] + taus*eta_sites[:,:,idi:idi+1])
            
            ids = self.events.metadata.index.to_frame().iloc[:,:].values
            subindex1 = np.repeat(ids, (nmaps),axis=0)
            subindex2 = np.tile(np.arange(nmaps).reshape(-1,1),(ids.shape[0],1))
            idnames = list(self.events.metadata.index.names)
            idnames.append('map_id')
            dfindex = pd.MultiIndex.from_arrays([subindex1[:,0],subindex1[:,1],subindex2.flatten()], names = idnames)
            
            df_im = pd.DataFrame(intensity.reshape((-1,nsites)),index=dfindex, columns = np.char.add('site', np.arange(self.events.sites.shape[0]).astype('str')))
            dfs_dict[intensity_id] = df_im
            if nmaps1==1:
                self.maps[intensity_id] = df_im.loc[:,:,:0]
            else:
                self.maps[intensity_id] = df_im
        return None

    def generate_hazard_curves(self,intensity_ids,imvals = np.logspace(-3,1,100)):
        intensity_ids = np.array(intensity_ids).reshape((-1,))
        for intensity_id in intensity_ids:
            imdf = self.maps[intensity_id]
            imvals_broad = np.repeat(imvals.reshape((-1,1)),imdf.shape[1],axis=1)
            rate_exceedance = np.zeros((imvals.shape[0],imdf.shape[1]))
            for source,rupture in self.events.metadata.index:
                imdf_event = imdf.loc[source,rupture,:]
                rate_exceedance+=imdf_event.apply(lambda x : 1 - ECDF(x)(imvals),axis=0).values*self.events.metadata.loc[source,rupture].annualized_rate
            self.hazard_curves[intensity_id] = (imvals_broad,rate_exceedance)

    def optimize_sa_maps(self,intensity_id,hazard_curves=None,
                         hz_curve_log10_range = [-4,-1.3],
                         log10_hz_curve_res = 0.05,
                         candies_per_event = 5,
                         candies_site_indices=None,
                         verbose=True):
                         # solver='SCS'
                         # eps=1e-7,eps_infeas=1e-9,
                         # max_iters= 500000):
        
        imdf = copy.deepcopy(self.maps[intensity_id])
        nsites = imdf.shape[1]
        
        if (hazard_curves is None) and ( intensity_id not in self.hazard_curves.keys()):
            self.generate_hazard_curves(intensity_id)
            im,hz_curves = self.hazard_curves[intensity_id]
        elif (hazard_curves is None) and ( intensity_id in self.hazard_curves.keys()):
            im,hz_curves = self.hazard_curves[intensity_id]
        else:
            im,hz_curves = hazard_curves

        hz_curves_min,hz_curves_max = hz_curve_log10_range
        
        curve_points = 10**np.arange(hz_curves_min,hz_curves_max+log10_hz_curve_res,log10_hz_curve_res)

        ims = np.array([interp1d(hz_curves[:,i],im[:,i])(curve_points) for i in range(nsites)])

        imcandy = imdf.loc[:,:,:candies_per_event-1]
        if not candies_site_indices is None:
            imcandy = imcandy.iloc[:,candies_site_indices]
            ims = ims[candies_site_indices,:]
            nsites = candies_site_indices.shape[0]
            
        w0 = imcandy.join(self.events.metadata.loc[:,['annualized_rate']]).loc[:,'annualized_rate'].values/candies_per_event
        imcandy_ar = imcandy.values

        theta = np.zeros((nsites*curve_points.shape[0],imcandy.shape[0]))
        for i in range(nsites):
            theta[i*curve_points.shape[0]:(i+1)*curve_points.shape[0],:] = (ims[i:i+1,:] < imcandy_ar[:,i:i+1]).T*1

        lambdas_IR = np.diag(np.tile(curve_points**-1,nsites))

        N=imcandy.shape[0]
        # M=nsites*curve_points.shape[0]

        U = theta
        # z = cp.Variable(M)
        x = cp.Variable(N)
        objective = cp.Minimize(cp.norm(1 - lambdas_IR@U@x,1))
        constraints = [x>=0,cp.sum(x)<=w0.sum()]
        prob = cp.Problem(objective, constraints)
        # prob = cp.Problem(cp.Minimize(cp.sum(z)), [-z<=1 - lambdas_IR@U@x,
        #                                            1 - lambdas_IR@U@x<=z,
        #                                            x>=0,
        #                                            cp.sum(x)<=w0.sum()])
        # optimal_value = prob.solve(solver=solver,verbose=verbose,
        #                            eps=eps,
        #                            eps_infeas=eps_infeas,
        #                            max_iters= max_iters)
        # print("z=",z.value)
        optimal_value = prob.solve(verbose=verbose)
        print("x=",x.value)
        print("val=",optimal_value)

        self.optimization_output.intensity_df = imcandy
        self.optimization_output.original_weight = w0
        self.optimization_output.optimal_weight = x.value
        self.optimization_output.interpolated_ims = ims
        self.optimization_output.interpolated_return_periods = curve_points
        
        return None