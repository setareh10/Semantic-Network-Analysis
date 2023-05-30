#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:07:29 2020

@author: sr05
"""

import mne
# import os
import numpy as np
from mne import make_forward_solution
import sn_config as C


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects_mri


for i in np.arange(0, len(subjects)):
    meg = '/'+subjects[i][5:]
    mri = subjects[i][1:15]
    
    # Path to input arguments of the forward_solution
    fname_bem_EMEG = data_path + meg + 'ico4_EEGMEG-bem.fif'
    fname_bem_MEG = data_path + meg + 'ico4_MEG-bem.fif'
    fname_trans = data_path + meg + 'MRI_'+ meg[7:11]+'-trans.fif'
    fname_src = data_path + meg + 'bem_oct6-src.fif'
    
    # complete path to raw data 
    raw_fname_SD  = data_path + meg + 'block_SD_tsss_raw.fif'

    ##..........................One-shell BEM for MEG................>.....##
    # Calculate a forward solution for a subject

    fwd_MEG = make_forward_solution(raw_fname_SD,trans=fname_trans, 
                  bem = fname_bem_MEG , meg=True, eeg=False, ignore_ref=False,
                  src= fname_src,n_jobs=1,verbose=None)
   
    # Path to save the bem file
    fname_MEG  = data_path + meg + 'block_MEG_fwd.fif'
  
    # Saving the bem file
    mne.write_forward_solution(fname_MEG ,fwd_MEG ,overwrite =True)

    
     ##..........................Three-shell BEM for MEG.....................##
    # Calculate a forward solution for a subject
    fwd_EMEG = make_forward_solution(raw_fname_SD,trans=fname_trans, 
                  bem = fname_bem_EMEG , meg=True, eeg=True, ignore_ref=False,
                  src= fname_src,n_jobs=1,verbose=None)
    
    # Path to save the bem file
    fname_EMEG  = data_path + meg + 'block_EMEG_fwd.fif'
 
    # Saving the bem file
    mne.write_forward_solution(fname_EMEG ,fwd_EMEG ,overwrite =True)
