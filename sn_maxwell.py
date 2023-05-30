#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:48:40 2020

@author: sr05
"""


import mne
import numpy as np
import os
import sn_config as C
from mne.preprocessing import find_bad_channels_maxwell
from mne.preprocessing import maxwell_filter


# path to unmaxfiltered raw data
data_path = C.data_path

# Path to a FIF file containing a MEG device head transformation
CBU_path = C.cbu_path

# Path to the FIF file with cross-talk correction information.
cross_talk = C.cross_talk

# Path to the '.dat' file with fine calibration coefficients
calibration = C.calibration

# subjects' directories
subjects =  C.subjects
destination_files = C.destination_files

for i in np.arange(0, len(subjects)):
    print('Participants : ',i)
    meg = subjects[i]

    # complete path to raw data 
    raw_fname_fruit = data_path + meg + 'block_fruit_fixed_raw.fif'  
    raw_fname_odour = data_path + meg + 'block_odour_fixed_raw.fif'
    raw_fname_milk  = data_path + meg + 'block_milk_fixed_raw.fif'
    raw_fname_LD    = data_path + meg + 'block_LD_fixed_raw.fif'

    
    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)
    raw_LD    = mne.io.Raw(raw_fname_LD   , preload=True)

    

    raw_fruit.info['bads'] = []
    raw_odour.info['bads'] = []
    raw_milk.info['bads']  = []
    raw_LD.info['bads']    = []
    
    # For closer equivalence with MaxFilter, it's recommended to low-pass 
    # filter your data prior to running this function
    raw_check_f = raw_fruit.copy().pick_types(exclude=()).load_data().filter(None,
                48)
    raw_check_o = raw_odour.copy().pick_types(exclude=()).load_data().filter(None,
                48)
    raw_check_m = raw_milk.copy().pick_types(exclude=()).load_data().filter(None,
                48)
    raw_check_l = raw_LD.copy().pick_types(exclude=()).load_data().filter(None,
                48)
    # Find MEG bad channels using Maxwell filtering
    auto_noisy_chs_f, auto_flat_chs_f = find_bad_channels_maxwell(raw_check_f,
                cross_talk = cross_talk, calibration = calibration,verbose=True)
    auto_noisy_chs_o, auto_flat_chs_o = find_bad_channels_maxwell(raw_check_o,
                cross_talk = cross_talk, calibration = calibration,verbose=True)
    auto_noisy_chs_m, auto_flat_chs_m = find_bad_channels_maxwell(raw_check_m,
                cross_talk = cross_talk, calibration = calibration,verbose=True)
    auto_noisy_chs_l, auto_flat_chs_l = find_bad_channels_maxwell(raw_check_l,
                cross_talk = cross_talk, calibration = calibration,verbose=True)

 
    raw_fruit.info['bads'].extend(auto_noisy_chs_f + auto_flat_chs_f)
    raw_odour.info['bads'].extend(auto_noisy_chs_o + auto_flat_chs_o)
    raw_milk.info['bads'].extend(auto_noisy_chs_m + auto_flat_chs_m)
    raw_LD.info['bads'].extend(auto_noisy_chs_l + auto_flat_chs_l)


    # # Making a dictionary of MEG bad channels
    # C.MEG_bad_channels_fruit[meg[1:11]] = auto_noisy_chs_f + auto_flat_chs_f 
    # C.MEG_bad_channels_odour[meg[1:11]] = auto_noisy_chs_o + auto_flat_chs_o  
    # C.MEG_bad_channels_milk[meg[1:11]]  = auto_noisy_chs_m + auto_flat_chs_m 
    # C.MEG_bad_channels_LD[meg[1:11]]    = auto_noisy_chs_l + auto_flat_chs_l 

    
    # # Saving MEG bad channels into a csv file
    # with open('MEG_bad_channels_fruit.csv', 'a') as f:
    #     f.write('{},{}\n'.format(meg[1:11],auto_noisy_chs_f + auto_flat_chs_f))
    #     f.close()
    # with open('MEG_bad_channels_odour.csv', 'a') as f:
    #     f.write('{},{}\n'.format(meg[1:11],auto_noisy_chs_o + auto_flat_chs_o))
    #     f.close()
    # with open('MEG_bad_channels_milk.csv', 'a') as f:
    #     f.write('{},{}\n'.format(meg[1:11],auto_noisy_chs_m + auto_flat_chs_m))
    #     f.close()
    # with open('MEG_bad_channels_LD.csv', 'a') as f:
    #     f.write('{},{}\n'.format(meg[1:11],auto_noisy_chs_l + auto_flat_chs_l))
    #     f.close()
        
    # Maxwell filtering data 
    raw_sss_fruit = maxwell_filter(raw_fruit, calibration = calibration, 
                    cross_talk=cross_talk , destination = CBU_path + meg
                    + destination_files[i])
    raw_sss_odour = maxwell_filter(raw_odour, calibration = calibration, 
                    cross_talk=cross_talk , destination = CBU_path + meg
                    + destination_files[i])
    raw_sss_milk  = maxwell_filter(raw_milk, calibration = calibration, 
                    cross_talk=cross_talk , destination = CBU_path + meg
                    + destination_files[i])
    raw_sss_LD    = maxwell_filter(raw_LD , calibration = calibration, 
                    cross_talk=cross_talk , destination = CBU_path + meg
                    + destination_files[i])
    
     # Checking for the desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)
   
    out_name_fruit = data_path + meg + 'block_fruit_tsss_raw.fif'
    out_name_odour = data_path + meg + 'block_odour_tsss_raw.fif'
    out_name_milk  = data_path + meg + 'block_milk_tsss_raw.fif'
    out_name_LD    = data_path + meg + 'block_LD_tsss_raw.fif'



    # Saving maxwellfiltered data     
    raw_sss_fruit.save(out_name_fruit, overwrite=True)
    raw_sss_odour.save(out_name_odour, overwrite=True)
    raw_sss_milk.save(out_name_milk, overwrite=True)
    raw_sss_LD.save(out_name_LD, overwrite=True)

