#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:16:31 2020

@author: sr05
"""
import mne
import numpy as np
import sn_config as C



# path to unmaxfiltered raw data
data_path = C.data_path

# subjects' directories
subjects =  C.subjects

for i in np.arange(0, len(subjects)):
    meg = subjects[i]

    raw_fname1 = data_path + meg + 'block_ld1_raw.fif'
    raw_fname2 = data_path + meg + 'block_ld2_raw.fif'
    raw1 = mne.io.Raw(raw_fname1, preload=True)
    raw2 = mne.io.Raw(raw_fname2, preload=True)
    raws=[raw1,raw2]
    
    events1 = mne.find_events(raw1,stim_channel='STI101',min_duration=0.001,
              shortest_event=1)  
    events2 = mne.find_events(raw2,stim_channel='STI101',min_duration=0.001,
              shortest_event=1)    
    eventss = [events1,events2]
    
    raw,events = mne.concatenate_raws(raws , events_list = eventss)
    
    raw_fname = data_path + meg + 'block_LD_raw.fif'
    raw.save(raw_fname, overwrite=True)
     
     
     
## only for subj 0125's milk block
for i in np.arange(18,19):

     meg = meg = subjects[i]
     
     raw_fname1 = data_path + meg + 'block_milk_1_raw.fif'
     raw_fname2 = data_path + meg + 'block_milk_2_raw.fif'
     raw1p = mne.io.Raw(raw_fname1, preload=True)
     raw1=raw1p.crop(tmin=0.0,tmax=169.0)
     raw2 = mne.io.Raw(raw_fname2, preload=True)
     raws=[raw1,raw2]
     
    
     events1 = mne.find_events(raw1, stim_channel='STI101', min_duration=0.001,
               shortest_event=1)  
     events2 = mne.find_events(raw2, stim_channel='STI101', min_duration=0.001,
               shortest_event=1) 
     eventss=[events1,events2]
     
     raw,events=mne.concatenate_raws(raws,events_list=eventss)
     raw_fname = data_path + meg + 'block_milk_raw.fif'
     raw.save(raw_fname, overwrite=True)
	