#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:04:45 2020

@author: sr05
"""

import mne
import numpy as np
import mne.viz 
from mayavi import mlab
import sn_config as C

# path to maxfiltered raw data
data_path = C.data_path

# subjects' directories
subjects_trans = C.subjects_trans

subjects =  C.subjects
subjects_MRI =  C.subjects_mri_files


for i in np.arange(0, len(subjects)-18):
    
    MRI = subjects_trans[i]
    meg = subjects[i]
    subjects_dir = data_path + meg + MRI
    raw = mne.io.read_raw_fif(data_path + meg + 'block_milk_fixed_raw.fif')

    mne.viz.plot_sensors(raw.info, kind='3d')
    
    trans = mne.read_trans(subjects_dir)
    # src = mne.read_source_spaces(src_fname)
    
    mne.viz.plot_alignment(raw.info, trans=trans, subject=subjects_MRI[i], 
            subjects_dir=subjects_dir, surfaces='brain', coord_frame='head')

    
    # fig_trans = plot_trans(info, trans_fname, subjects_dir=C.subjects_dir, subject=subject,
    #             dig=True, meg_sensors=True, eeg_sensors='original', src=src)
    