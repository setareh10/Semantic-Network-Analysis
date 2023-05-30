#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:55:25 2020

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
subjects_MRI = C.subjects_mri
subjects = C.subjects


print("Please enter the participant number (in the range of : [0,18] ):\n")
i = input("The participant number is:\n")
if int(i) not in np.arange(0,19):
     print('ERROR!')
     print("Please enter the participant number (in the range of : [0,18] ):\n")
     i = input("The participant number is:\n")
i=int(i)
subject_from = subjects[i]
meg = subjects[i]   
print('participant : ' , meg[1:11] )


# loading raw data
raw_fname = data_path + subjects[i] + 'block_fruit_tsss_raw.fif'
raw = mne.io.Raw(raw_fname, preload=True)

trans = data_path + subjects[i] + subjects_trans[i]

fig = mne.viz.plot_alignment(raw.info, trans=trans,
      subject = data_path + subjects_MRI[i][0:16], subjects_dir=data_path,
      surfaces=['head'], eeg=['original', 'projected'], meg='sensors')
    
    