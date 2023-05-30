#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:33:02 2020

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2

for win in np.arange(3, len(C.time_window)):
    t_min_crop= C.time_window[win]
    t_max_crop= C.time_window[win] + C.time_window_len
# t_min_crop= 0.050
# t_max_crop= 0.600
    for i in np.arange(0, len(subjects)):
        n_subjects = len(subjects)
        meg = subjects[i]
        # print('Participant : ' , i, '/ win : ',win)
        
        # Reading epochs
        epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
        epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
            
        epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
        epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
        
        epochs_SD = epochs_sd['words'] 
        epochs_LD = epochs_ld['words'] 
    
        # Equalize trial counts to eliminate bias (which would otherwise be
        # introduced by the abs() performed below)
        # equalize_epoch_counts([epochs_SD, epochs_LD])
        
        # Reading inverse operator
        inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
        inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
    
        inv_op_SD = read_inverse_operator(inv_fname_SD) 
        inv_op_LD = read_inverse_operator(inv_fname_LD) 
        
        # Evoked responses 
        evoked_SD = epochs_SD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
        evoked_LD = epochs_LD.average().set_eeg_reference(ref_channels = \
                            'average',projection=True)
            
        # Applying inverse solution to get sourse signals       
        stc_SD = apply_inverse(evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                               pick_ori=None)
        stc_LD = apply_inverse(evoked_LD, inv_op_LD,lambda2,method ='MNE',
                               pick_ori=None)
        
        # Averaging sourse signals across a time window :[0.050:0.100:0.550]
        stc_SD_mean = stc_SD.copy().crop(t_min_crop, t_max_crop).mean()
        stc_LD_mean = stc_LD.copy().crop(t_min_crop, t_max_crop).mean()
        tmin = stc_SD_mean.tmin
        tstep = stc_SD_mean.tstep
    
        # Morphing source signals onto fsaverage
        morph_SD = mne.compute_source_morph( src= inv_op_SD['src'],subject_from\
                   = stc_SD.subject , subject_to = C.subject_to , spacing = \
                   C.spacing_morph, subjects_dir = C.data_path)    
        morph_LD = mne.compute_source_morph( src= inv_op_LD['src'],subject_from\
                   = stc_LD.subject , subject_to = C.subject_to , spacing = \
                   C.spacing_morph, subjects_dir = C.data_path) 
        
        stc_fsaverage_SD = morph_SD.apply(stc_SD_mean)
        stc_fsaverage_LD = morph_LD.apply(stc_LD_mean)
        stc_fsaverage = stc_fsaverage_SD - stc_fsaverage_LD
        
        if i==0:
            stc_t_SD = stc_fsaverage_SD
            stc_t_LD = stc_fsaverage_LD
            stc_t = stc_fsaverage

        else:
            stc_t_SD = stc_t_SD+ stc_fsaverage_SD
            stc_t_LD = stc_t_LD+ stc_fsaverage_LD
            stc_t = stc_t+ stc_fsaverage
            
    X=stc_t_SD.copy()/18
    Y=stc_t_LD.copy()/18
    Z=X-Y
    thresh_max=max(X.data.max(),Y.data.max())
    thresh_min=max(X.data.min(),Y.data.min())
    
    brain = X.plot(surface='inflated', hemi='split',subject =\
                'fsaverage',  subjects_dir=data_path,size=(800,400),initial_time=0.100,time_unit='s',
                clim=dict(kind='value', lims= [thresh_min,(thresh_min+thresh_max)/4,thresh_max]))
    brain.save_image(C.pictures_path_Source_estimate+'Evoked Responses_SD_'+\
                 f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}.png')
 
    brain = Y.plot(surface='inflated', hemi='split',subject =\
                'fsaverage',  subjects_dir=data_path,size=(800,400),initial_time=0.100,time_unit='s',
                clim=dict(kind='value', lims= [thresh_min,(thresh_min+thresh_max)/4,thresh_max]))    
    brain.save_image(C.pictures_path_Source_estimate+'Evoked Responses_LD_'+\
                 f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}.png')
     
    brain = Z.plot(surface='inflated', hemi='split',subject =\
                'fsaverage',  subjects_dir=data_path,size=(800,400),initial_time=0.100,time_unit='s',
                clim=dict(kind='percent', pos_lims= [50,75,100]))      
    brain.save_image(C.pictures_path_Source_estimate+'Evoked Responses_SD-LD_'+\
                 f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}.png')
          
                
#     stc_total = stc_t/len(subjects) 
#     C.stc_all.append(stc_total)
#     idx = stc_total.time_as_index(times=stc_total.times)
#     data = stc_total.data[:, idx]
#     C.min_max_val.append([data.min() , data.max()])

# max_val = max(max(C.min_max_val))
# min_val = min(min(min(C.min_max_val)),0)
# thresh= max(abs(max_val),abs(min_val))


# max_val = 1.32e-11
# min_val = -6.41e-12
# mid_val = (max_val  - min_val)/2

# for n in np.arange(0, len(C.stc_all)): 
#     t_min_crop= C.time_window[n]
#     t_max_crop= C.time_window[n] + C.time_window_len
#     # brain = C.stc_all[n].plot(surface='inflated', hemi='split',subject =\
#     #         'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', lims=\
#     #         [min_val , mid_val ,max_val]),size=(800,400))
#       # brain = C.stc_all[n].plot(surface='inflated', hemi='split',subject =\
#       #       'fsaverage',  subjects_dir=data_path,size=(800,400),colormap='mne')
    
    
#     # brain = C.stc_all[n].plot(surface='inflated', hemi='split',subject =\
#     #         'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', lims=\
#     #         [min_val , min_val/10  ,max_val]),size=(800,400),colormap='mne')

#     brain = C.stc_all[n].plot(surface='inflated', hemi='split',subject =\
#             'fsaverage',  subjects_dir=data_path, clim=dict(kind='value', lims=\
#             [-thresh , 0  ,thresh]),size=(800,400),colormap='mne')

#     # brain.save_image(C.pictures_path_Source_estimate+'Evoked Responses_'+\
#     #                   f'{t_min_crop:.3f}' +'_'+f'{t_max_crop:.3f}_nonequalized.png')
 
# kind='percent', pos_lims= [50,75,100]))    