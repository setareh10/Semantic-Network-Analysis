
"""
Created on Thu Mar  5 19:16:48 2020

@author: sr05
"""
import mne
import os
import numpy as np
import sn_config as C

# path to maxfiltered raw data
data_path = C.data_path

# subjects' directories
subjects = C.subjects


for i in np.arange(0,len(subjects)):

    meg = subjects[i]  
    # complete path to raw data 
    raw_fname_fruit = data_path + meg + 'block_fruit_raw.fif'
    raw_fname_odour = data_path + meg + 'block_odour_raw.fif'
    raw_fname_milk  = data_path + meg + 'block_milk_raw.fif'
    raw_fname_LD    = data_path + meg + 'block_LD_raw.fif'
    
    check_cmd_fruit = '/imaging/local/software/mne/mne_2.7.3/x86_64/\
    MNE-2.7.3-3268-Linux-x86_64/bin/mne_check_eeg_locations \
    --file '+ raw_fname_fruit +' --fix'
    
    check_cmd_odour = '/imaging/local/software/mne/mne_2.7.3/x86_64/\
    MNE-2.7.3-3268-Linux-x86_64/bin/mne_check_eeg_locations \
    --file '+ raw_fname_odour +' --fix'
    
    check_cmd_milk = '/imaging/local/software/mne/mne_2.7.3/x86_64/\
    MNE-2.7.3-3268-Linux-x86_64/bin/mne_check_eeg_locations \
    --file '+ raw_fname_milk +' --fix'
    
    check_cmd_LD = '/imaging/local/software/mne/mne_2.7.3/x86_64/\
    MNE-2.7.3-3268-Linux-x86_64/bin/mne_check_eeg_locations \
    --file '+ raw_fname_LD +' --fix'
    
    
    os.system(check_cmd_fruit)
    os.system(check_cmd_odour)
    os.system(check_cmd_milk)
    os.system(check_cmd_LD)
    

    # loading raw data
    raw_fruit_fixed = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour_fixed = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk_fixed  = mne.io.Raw(raw_fname_milk , preload=True)
    raw_LD_fixed    = mne.io.Raw(raw_fname_LD   , preload=True)

  
    # checking for the desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    
    out_name_fruit = data_path + meg + 'block_fruit_fixed_raw.fif'
    out_name_odour = data_path + meg + 'block_odour_fixed_raw.fif'
    out_name_milk  = data_path + meg + 'block_milk_fixed_raw.fif'
    out_name_LD    = data_path + meg + 'block_LD_fixed_raw.fif'

	
    # saving filtered data
         
    raw_fruit_fixed.save(out_name_fruit, overwrite=True)
    raw_odour_fixed.save(out_name_odour, overwrite=True)
    raw_milk_fixed.save(out_name_milk, overwrite=True)
    raw_LD_fixed.save(out_name_LD, overwrite=True)



