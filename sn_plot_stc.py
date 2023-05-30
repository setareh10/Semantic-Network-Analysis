#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:47:10 2020

@author: sr05
"""


import mne
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
import os


# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
pictures_path = C.pictures_path_Source_estimate


# subjects' directories
subjects =  C.subjects_mri

print("Please enter the participant number (in the range of : [0,18] ):\n")
i = input("The participant number is:\n")
if int(i) not in np.arange(0,19):
     print('ERROR!')
     print("Please enter the participant number (in the range of : [0,18] ):\n")
     i = input("The participant number is:\n")
i=int(i)
subject_from = subjects[i]
meg = subjects[i][5:]    
print('participant : ' , meg[0:10] )


print("Please enter your block from the list bellow:\n")
print(C.block_names)
value1 = input("Your selected block is:\n")

if value1==('fruit' or 'odour' or 'milk'):
    print("Please enter your categories from the list bellow:\n")
    print(C.categories_sd)
    value2 = input("Your selected category is:\n") 
    if (value2 not in C.categories_sd or value2== 'words'):
        print('ERROR!')
        print("Please enter your categories from the list bellow:\n")
        print(C.categories_sd)
        value2 = input("Your selected category is:\n") 
        
if value1=='LD':
    print("Please enter your categories from the list bellow:\n")
    print(C.categories_ld)
    value2 = input("Your selected category is:\n")
    if value2 not in C.categories_ld:
        print('ERROR!')
        print("Please enter your categories from the list bellow:\n")
        print(C.categories_ld)
        value2 = input("Your selected category is:\n")
    
if value1=='SD':
    print("The only possible category is 'words':\n")   
    value2 = 'words'   
    


fname_fsaverage = C.data_path + meg + 'block_'+value1+'_'+value2+'_EMEG_fsaverage'
stc= mne.read_source_estimate(fname_fsaverage )

# get index for time point to plot
idx = stc.time_as_index(times=stc.times)
data = stc.data[:, idx]
thresh = np.abs(data).max()     
vertno_max, time_max = stc.get_peak(hemi='lh')
stc.plot(surface='inflated', hemi='lh',subject = 'fsaverage', 
         subjects_dir=data_path, clim=dict(kind='value',lims=[0, thresh/ 20.,
         thresh]),initial_time= time_max )
