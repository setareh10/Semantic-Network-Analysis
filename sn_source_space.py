#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 21:46:16 2020

@author: sr05
"""


import mne
import numpy as np
import sn_config as C


# path to data
data_path = C.data_path

subjects =  C.subjects_mri
src_spacing = C.src_spacing
for i in np.arange(0, len(subjects)):
    print('Participant : ' , i)
    meg = subjects[i][5:]
    mri = subjects[i][1:15]
    
    # Set up bilateral hemisphere surface-based source space with subsampling
    src = mne.setup_source_space(subject = mri, spacing = src_spacing,
          subjects_dir = data_path, add_dist = True)
    # Path to save the file
    src_fname = data_path + meg + 'bem_oct6-src.fif'
    # Saving the file
    mne.write_source_spaces(src_fname, src, overwrite=True)