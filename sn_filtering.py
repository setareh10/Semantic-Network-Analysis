"""
Created on Thu Mar  5 19:16:48 2020

@author: sr05
"""

import mne
import os
import numpy as np
import sn_config as c

# path to maxfiltered raw data
main_path = c.main_path
data_path = c.data_path

# filter parameters
l_freq = c.l_freq
h_freq = c.h_freq

# subjects' directories
subjects = c.subjects

# EEG bad channels
eeg_bad_channels_fruit = c.eeg_bad_channels_fruit
eeg_bad_channels_odour = c.eeg_bad_channels_odour
eeg_bad_channels_milk = c.eeg_bad_channels_milk
eeg_bad_channels_ld = c.eeg_bad_channels_ld

# MEG bad channels
meg_bad_channels_fruit = c.meg_bad_channels_fruit
meg_bad_channels_odour = c.meg_bad_channels_odour
meg_bad_channels_milk = c.meg_bad_channels_milk
meg_bad_channels_ld = c.meg_bad_channels_ld

for i in np.arange(len(subjects)):
    print(f'Participants: {i}')
    meg = subjects[i]

    # complete path to raw data 
    raw_fname_fruit = main_path + meg + 'block_fruit_tsss_raw.fif'
    raw_fname_odour = main_path + meg + 'block_odour_tsss_raw.fif'
    raw_fname_milk = main_path + meg + 'block_milk_tsss_raw.fif'
    raw_fname_ld = main_path + meg + 'block_ld_tsss_raw.fif'

    # loading raw data
    raw_fruit = mne.io.Raw(raw_fname_fruit, preload=True)
    raw_odour = mne.io.Raw(raw_fname_odour, preload=True)
    raw_milk = mne.io.Raw(raw_fname_milk, preload=True)
    raw_ld = mne.io.Raw(raw_fname_ld, preload=True)

    # Adding MEG bad channels to raw.info 
    raw_fruit.info['bads'].extend(meg_bad_channels_fruit[i] +
                                  eeg_bad_channels_fruit[i])
    raw_odour.info['bads'].extend(meg_bad_channels_odour[i] +
                                  eeg_bad_channels_odour[i])
    raw_milk.info['bads'].extend(meg_bad_channels_milk[i] +
                                 eeg_bad_channels_milk[i])
    raw_ld.info['bads'].extend(meg_bad_channels_ld[i] +
                               eeg_bad_channels_ld[i])

    # Interpolating MEG bad channels
    raw_fruit.interpolate_bads(reset_bads=True, mode='fast')
    raw_odour.interpolate_bads(reset_bads=True, mode='fast')
    raw_milk.interpolate_bads(reset_bads=True, mode='fast')
    raw_ld.interpolate_bads(reset_bads=True, mode='fast')

    # re-referencing the data according to the desired reference
    raw_fruit.set_eeg_reference(ref_channels='average')
    raw_odour.set_eeg_reference(ref_channels='average')
    raw_milk.set_eeg_reference(ref_channels='average')
    raw_ld.set_eeg_reference(ref_channels='average')

    picks_fruit = mne.pick_types(raw_fruit.info, meg=True, eeg=True, eog=False,
                                 stim=False)
    picks_odour = mne.pick_types(raw_odour.info, meg=True, eeg=True, eog=False,
                                 stim=False)
    picks_milk = mne.pick_types(raw_milk.info, meg=True, eeg=True, eog=False,
                                stim=False)
    picks_ld = mne.pick_types(raw_ld.info, meg=True, eeg=True, eog=False,
                              stim=False)

    # Notch filter for the raw data
    raw_fruit_notch = raw_fruit.copy().notch_filter(freqs=50,
                                                    picks=picks_fruit)
    raw_odour_notch = raw_odour.copy().notch_filter(freqs=50,
                                                    picks=picks_odour)
    raw_milk_notch = raw_milk.copy().notch_filter(freqs=50,
                                                  picks=picks_milk)
    raw_ld_notch = raw_ld.copy().notch_filter(freqs=50,
                                              picks=picks_ld)

    # Band-pass filtering
    raw_fruit_notch_bpf = raw_fruit_notch.copy().filter(l_freq=l_freq,
                                                        h_freq=h_freq,
                                                        fir_design='firwin',
                                                        picks=picks_fruit)
    raw_odour_notch_bpf = raw_odour_notch.copy().filter(l_freq=l_freq,
                                                        h_freq=h_freq,
                                                        fir_design='firwin',
                                                        picks=picks_odour)
    raw_milk_notch_bpf = raw_milk_notch.copy().filter(l_freq=l_freq,
                                                      h_freq=h_freq,
                                                      fir_design='firwin',
                                                      picks=picks_milk)
    raw_ld_notch_bpf = raw_ld_notch.copy().filter(l_freq=l_freq,
                                                  h_freq=h_freq,
                                                  fir_design='firwin',
                                                  picks=picks_ld)

    # checking for the desired directory to save the data
    if not os.path.isdir(data_path + meg):
        os.makedirs(data_path + meg)

    out_name_fruit = data_path + meg + \
        'block_fruit_tsss_notch_bpf0.1_45_raw.fif'
    out_name_odour = data_path + meg + \
        'block_odour_tsss_notch_bpf0.1_45_raw.fif'
    out_name_milk = data_path + meg + \
        'block_milk_tsss_notch_bpf0.1_45_raw.fif'
    out_name_ld = data_path + meg + \
        'block_ld_tsss_notch_bpf0.1_45_raw.fif'

    # saving filtered data

    raw_fruit_notch_bpf.save(out_name_fruit, overwrite=True)
    raw_odour_notch_bpf.save(out_name_odour, overwrite=True)
    raw_milk_notch_bpf.save(out_name_milk, overwrite=True)
    raw_ld_notch_bpf.save(out_name_ld, overwrite=True)
