

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
subjects =  C.subjects


for i in np.arange(0, len(subjects)):
    meg = subjects[i]

    
    # complete path to raw data 
    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_fname_milk  = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_fname       = data_path + meg + 'block_SD_tsss_raw.fif'
       # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk  = mne.io.Raw(raw_fname_milk , preload=True)

    raw_SD = mne.concatenate_raws([raw_fruit,raw_odour,raw_milk ])
    raw_SD.save(raw_fname, overwrite=True)