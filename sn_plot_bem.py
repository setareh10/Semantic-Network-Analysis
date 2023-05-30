#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:26:42 2020

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
mne.viz.plot_bem( subject = data_path + subjects_MRI[i][0:16], 
                 subjects_dir=data_path , brain_surfaces='white',
                 orientation='coronal')
    
