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

from pyopensha.utils import haversine_distance

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
        site_df = pd.read_csv(self.files.user + '\\' + self.files.site_filename)
        self.output = self.site_output()
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
            jarpath = os.path.join("..", "pypsha", "jar", "IM_EventSetCalc_v3_0_ASCII.jar")
    
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