#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:09:46 2020

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

# bem parameters
bem_ico = C.bem_ico
conductivity_1 = C.conductivity_1
conductivity_3 = C.conductivity_3

subjects=  C.subjects

for i in np.arange(0,len(subjects)):
    meg = '/'+subjects[i][5:]
    mri = subjects[i][1:15]

    ## >>>>>>>>>>>>>>>>>>>>>>>One-shell BEM for MEG<<<<<<<<<<<<<<<<<<<<<<<<<##
 
    # Create a BEM model for a subject
    model = mne.make_bem_model(subject=mri, ico = bem_ico,
            conductivity = conductivity_1,subjects_dir = data_path )
    
    # Create a BEM solution using the linear collocation approach
    bem = mne.make_bem_solution(model)
    
    # Path to save the bem file
    bem_fname = data_path + meg + 'ico4_MEG-bem.fif'
    
    # Saving the file
    mne.bem.write_bem_solution(bem_fname, bem)

    ## >>>>>>>>>>>>>>>>>>>>Three-shell BEM for MEG<<<<<<<<<<<<<<<<<<<<<<<<<<##
    
    # Create a BEM model for a subject
    model = mne.make_bem_model(subject=mri, ico=bem_ico,
            conductivity = conductivity_3 , subjects_dir = data_path)
    
    # Create a BEM solution using the linear collocation approach
    bem = mne.make_bem_solution(model)

    # Path to save the bem file
    bem_fname = data_path + meg +'ico4_EEGMEG-bem.fif'

    # Saving the file
    mne.bem.write_bem_solution(bem_fname, bem)
    
    
    