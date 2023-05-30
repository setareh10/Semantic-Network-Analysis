#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:38:04 2020

@author: sr05
"""


import numpy as np
import mne
import sn_config as C

# Freesurfer and MNE environment variables
filename = "/imaging/local/software/mne_python/set_MNE_2.7.3_FS_6.0.0_environ.py"
# for Python 3 instead of execfile
exec(compile(open(filename, "rb").read(), filename, 'exec'))

# path to MRI data
data_path = C.data_path

subjects =  C.subjects_mri

for i in np.arange(0,len(subjects)):
    mri = subjects[i][1:15]

    # Creating BEM surfaces using the FreeSurfer watershed algorithm
    mne.bem.make_watershed_bem(subject = mri , subjects_dir = data_path , 
                               overwrite= True)

  