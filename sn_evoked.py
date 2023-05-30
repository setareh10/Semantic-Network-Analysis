
"""
Created on Fri Mar  6 14:29:15 2020

@author: sr05
"""

import mne
import os
import numpy as np
import sn_config as C


# path to maxfiltered raw data
data_path = C.data_path

# Events info
event_id_SD = C.event_id_sd
event_id_LD = C.event_id_ld
  
# Parameters
reject = C.reject
evoked_fruit_categories = dict()
evoked_odour_categories = dict()
evoked_milk_categories = dict()
evoked_LD_categories = dict()


# subjects' directories
subjects =  C.subjects

 
for i in np.arange(0, len(subjects)):

    meg = subjects[i]
    
    # Complete path to epoched data 
    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk  = data_path + meg + 'block_milk_epochs-epo.fif'
    epoch_fname_LD    = data_path + meg + 'block_LD_epochs-epo.fif'

    
    # Loading epoched data 
    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk  = mne.read_epochs(epoch_fname_milk , preload=True)
    epochs_LD    = mne.read_epochs(epoch_fname_LD   , preload=True)

    
    
    # Loading events 
    events_fruit = mne.read_events(data_path + meg + 'block_fruit-eve.fif')
    events_odour = mne.read_events(data_path + meg + 'block_odour-eve.fif')
    events_milk  = mne.read_events(data_path + meg + 'block_milk-eve.fif' )
    events_LD    = mne.read_events(data_path + meg + 'block_LD-eve.fif' )


    picks_fruit = mne.pick_types(epochs_fruit.info, meg=True, eeg=True)
    picks_odour = mne.pick_types(epochs_odour.info, meg=True, eeg=True)
    picks_milk  = mne.pick_types(epochs_milk.info , meg=True, eeg=True)
    picks_LD    = mne.pick_types(epochs_LD.info   , meg=True, eeg=True)

    # Computing evoked data 
    evoked_fruit = epochs_fruit.average(picks=picks_fruit)
    evoked_odour = epochs_odour.average(picks=picks_odour)
    evoked_milk  = epochs_milk.average(picks=picks_milk)
    evoked_LD    = epochs_LD.average(picks=picks_LD)


    # Computing evoked data for each word category
    evoked_fruit_visual   = epochs_fruit['visual'].average(picks=picks_fruit)
    evoked_fruit_hear     = epochs_fruit['hear'].average(picks=picks_fruit)
    evoked_fruit_hand     = epochs_fruit['hand'].average(picks=picks_fruit)
    evoked_fruit_neutral  = epochs_fruit['neutral'].average(picks=picks_fruit)
    evoked_fruit_emotional= epochs_fruit['emotional'].average(picks=picks_fruit)
    evoked_fruit_pwordc   = epochs_fruit['pwordc'].average(picks=picks_fruit)
    evoked_fruit_target   = epochs_fruit['target'].average(picks=picks_fruit)
   
    evoked_odour_visual   = epochs_odour['visual'].average(picks=picks_odour)
    evoked_odour_hear     = epochs_odour['hear'].average(picks=picks_odour)
    evoked_odour_hand     = epochs_odour['hand'].average(picks=picks_odour)
    evoked_odour_neutral  = epochs_odour['neutral'].average(picks=picks_odour)
    evoked_odour_emotional= epochs_odour['emotional'].average(picks=picks_odour)
    evoked_odour_pwordc   = epochs_odour['pwordc'].average(picks=picks_odour)
    evoked_odour_target   = epochs_odour['target'].average(picks=picks_odour)
    
    evoked_milk_visual    = epochs_milk['visual'].average(picks=picks_milk)
    evoked_milk_hear      = epochs_milk['hear'].average(picks=picks_milk)
    evoked_milk_hand      = epochs_milk['hand'].average(picks=picks_milk)
    evoked_milk_neutral   = epochs_milk['neutral'].average(picks=picks_milk)
    evoked_milk_emotional = epochs_milk['emotional'].average(picks=picks_milk)
    evoked_milk_pwordc    = epochs_milk['pwordc'].average(picks=picks_milk)
    evoked_milk_target    = epochs_milk['target'].average(picks=picks_milk)
    
    evoked_LD_visual    = epochs_LD['visual'].average(picks=picks_LD)
    evoked_LD_hear      = epochs_LD['hear'].average(picks=picks_LD)
    evoked_LD_hand      = epochs_LD['hand'].average(picks=picks_LD)
    evoked_LD_neutral   = epochs_LD['neutral'].average(picks=picks_LD)
    evoked_LD_emotional = epochs_LD['emotional'].average(picks=picks_LD)
    evoked_LD_pwordc    = epochs_LD['pwordc'].average(picks=picks_LD)
    evoked_LD_pworda    = epochs_LD['pwordc'].average(picks=picks_LD)
    evoked_LD_filler    = epochs_LD['filler'].average(picks=picks_LD)

    
    
    evoked_fruit_categories = {'visual':evoked_fruit_visual , 
            'hear':evoked_fruit_hear,'hand':evoked_fruit_hand, 
            'neutral':evoked_fruit_neutral,'emotional':evoked_fruit_emotional,
            'pwordc':evoked_fruit_pwordc,'target':evoked_fruit_target}
 
    evoked_odour_categories = {'visual':evoked_odour_visual , 
            'hear':evoked_odour_hear, 'hand':evoked_odour_hand, 
            'neutral':evoked_odour_neutral,'emotional':evoked_odour_emotional,
            'pwordc':evoked_odour_pwordc,'target':evoked_odour_target}
    
    evoked_milk_categories  = {'visual':evoked_milk_visual , 
            'hear':evoked_milk_hear,'hand':evoked_milk_hand, 
            'neutral':evoked_milk_neutral,'emotional':evoked_milk_emotional, 
            'pwordc':evoked_milk_pwordc,'target':evoked_milk_target}
    
    evoked_LD_categories  = {'visual':evoked_LD_visual , 
            'hear':evoked_LD_hear,'hand':evoked_LD_hand, 
            'neutral':evoked_LD_neutral,'emotional':evoked_LD_emotional, 
            'pwordc':evoked_LD_pwordc, 'pworda':evoked_LD_pworda,
            'filler':evoked_LD_filler}
    

    # checking for the existance of desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    out_name_fruit = data_path + meg + 'block_fruit_evoked-ave.fif'
    out_name_odour = data_path + meg + 'block_odour_evoked-ave.fif'
    out_name_milk  = data_path + meg + 'block_milk_evoked-ave.fif'
    out_name_LD    = data_path + meg + 'block_LD_evoked-ave.fif'

    
    # saving evoked data
    mne.write_evokeds(out_name_fruit , evoked_fruit)
    mne.write_evokeds(out_name_odour , evoked_odour)
    mne.write_evokeds(out_name_milk  , evoked_milk)
    mne.write_evokeds(out_name_LD    , evoked_LD)


    out_name_f_c = data_path + meg + 'block_fruit_evoked_categories.npy'
    out_name_o_c = data_path + meg + 'block_odour_evoked_categories.npy'
    out_name_m_c = data_path + meg + 'block_milk_evoked_categories.npy'
    out_name_l_c = data_path + meg + 'block_LD_evoked_categories.npy'

    # saving the evoked of categories as a dictionary    
    np.save(out_name_f_c, evoked_fruit_categories)
    np.save(out_name_o_c, evoked_odour_categories)
    np.save(out_name_m_c, evoked_milk_categories)
    np.save(out_name_l_c, evoked_LD_categories)

    

    
    




