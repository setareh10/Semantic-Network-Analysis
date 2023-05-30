#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:36:05 2020

@author: sr05
"""
# import sys
# sys.path.append('/imaging/local/software/miniconda/lib/python3.7/site-packages/')
import numpy as np
import mne
import sn_config as C

data_path = C.data_path
# Freesurfer and MNE environment variables
filename = "/imaging/local/software/mne_python/set_MNE_2.7.3_FS_6.0.0_environ.py"
# for Python 3 instead of execfile
exec(compile(open(filename, "rb").read(), filename, 'exec'))
 
subjects =  C.subjects_mri
    
for i in np.arange(0,len(subjects)-17):

    meg = '/' + subjects[i][5:]
    mri = subjects[i][1:15]
    subject = mri
    subjects_dir = data_path 
    # Path to an instance file containing the digitizer data 
    raw_fname = data_path + meg + 'block_fruit_tsss_raw.fif'
    
    # Coregister an MRI with a subject's head shape
    mne.gui.coregistration(inst = raw_fname, subject =  subject, 
            subjects_dir =  subjects_dir , advanced_rendering=False)
  
