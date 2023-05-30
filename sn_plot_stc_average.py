#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 08:30:01 2020

@author: sr05
"""


import mne
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
import os


# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
pictures_path = C.pictures_path_Source_estimate


# subjects' directories
subjects =  C.subjects

for i in np.arange(0,len(subjects)):
    print('Participant : ', i)
    meg = subjects[i]  

    
    fname_fsaverage_SD = C.data_path + meg + 'block_SD_words_EMEG_fsaverage'
    fname_fsaverage_LD = C.data_path + meg + 'block_LD_words_EMEG_fsaverage'
    
    stc_SD= mne.read_source_estimate(fname_fsaverage_SD )
    stc_LD= mne.read_source_estimate(fname_fsaverage_LD )
    
    if i==0:
        stc_SD_all = stc_SD
        stc_LD_all = stc_LD
    else:
        stc_SD_all = stc_SD_all + stc_SD
        stc_LD_all = stc_LD_all + stc_LD
        
stc_SD_all =  stc_SD_all/len(subjects)
stc_LD_all =  stc_LD_all/len(subjects)

# get index for time point to plot
idx_SD = stc_SD_all.time_as_index(times=stc_SD_all.times)
data_SD = stc_SD_all.data[:, idx_SD]
thresh_SD = np.abs(data_SD).max()     
vertno_max_SD, time_max_SD = stc_SD_all.get_peak(hemi='lh')

int_time= 0.400
brain_SD = stc_SD_all.plot(surface='inflated', hemi='split',subject = 'fsaverage', 
          subjects_dir=data_path, clim=dict(kind='value',lims=[0, thresh_SD/ 5.,
          thresh_SD]),initial_time= int_time,size=(800,400) )

brain_SD.save_image(C.pictures_path_Source_estimate+'Evoked Responses_SD'+\
                      f'{int_time:.3f}.png')

idx_LD = stc_LD_all.time_as_index(times=stc_LD_all.times)
data_LD = stc_LD_all.data[:, idx_LD]
thresh_LD = np.abs(data_LD).max()     
vertno_max_LD, time_max_LD = stc_LD_all.get_peak(hemi='lh')

int_time= 0.284
brain_LD = stc_LD_all.plot(surface='inflated', hemi='split',subject = 'fsaverage', 
          subjects_dir=data_path, clim=dict(kind='value',lims=[0, thresh_SD/ 5.,
          thresh_SD]),initial_time= int_time,size=(800,400) )
brain_LD.save_image(C.pictures_path_Source_estimate+'Evoked Responses_LD'+\
                      f'{int_time:.3f}.png')