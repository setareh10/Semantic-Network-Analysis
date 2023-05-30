#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:59:30 2020

@author: sr05
"""

import os
import pickle
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity, seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test, spatio_temporal_cluster_test,
                       summarize_clusters_stc, permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time
import sys

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr_epoch
lambda2 = C.lambda2_epoch
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()
n_subjects = len(subjects)
stc_total_F = [[[0]*4 for k in range(6)] for w in range(2)]
stc_total_M = [[[0]*4 for k in range(6)] for w in range(2)]
stc_total_O = [[[0]*4 for k in range(6)] for w in range(2)]
stc_total_SD = [[[0]*4 for k in range(6)] for w in range(2)]
stc_total_LD = [[[0]*4 for k in range(6)] for w in range(2)]

method = 'coh'


def SN_functional_connectivity_bands_runs(i, method, SN_ROI):
    s = time.time()
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    stc_F_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_F_bands_SD_sub'+str(i)+'.json'
    stc_O_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_O_bands_LD_sub'+str(i)+'.json'
    stc_M_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_M_bands_SD_sub'+str(i)+'.json'
    stc_SD_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_mean_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_mean_bands_LD_sub'+str(i)+'.json'

    morphed_labels = mne.morph_labels(SN_ROI, subject_to=sub_to,
                                      subject_from='fsaverage', subjects_dir=data_path)

    # Reading epochs

    # Reading epochs
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'

    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

    epochs_LD = epochs_ld['words'].copy().resample(500)

    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk = data_path + meg + 'block_milk_epochs-epo.fif'

    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk = mne.read_epochs(epoch_fname_milk, preload=True)

    epochs_f = mne.epochs.combine_event_ids(epochs_fruit, ['visual',
                                                           'hear', 'hand', 'neutral', 'emotional'], {'words': 15})
    epochs_o = mne.epochs.combine_event_ids(epochs_odour, ['visual',
                                                           'hear', 'hand', 'neutral', 'emotional'], {'words': 15})
    epochs_m = mne.epochs.combine_event_ids(epochs_milk, ['visual',
                                                          'hear', 'hand', 'neutral', 'emotional'], {'words': 15})

    epochs_f = epochs_f['words'].copy().resample(500)
    epochs_o = epochs_o['words'].copy().resample(500)
    epochs_m = epochs_m['words'].copy().resample(500)

    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD)
    inv_op_LD = read_inverse_operator(inv_fname_LD)

    stc_f = apply_inverse_epochs(epochs_f, inv_op_SD, lambda2, method='MNE',
                                 pick_ori="normal", return_generator=False)
    stc_o = apply_inverse_epochs(epochs_o, inv_op_SD, lambda2, method='MNE',
                                 pick_ori="normal", return_generator=False)
    stc_m = apply_inverse_epochs(epochs_m, inv_op_SD, lambda2, method='MNE',
                                 pick_ori="normal", return_generator=False)
    stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD, lambda2, method='MNE',
                                  pick_ori="normal", return_generator=False)

    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']

    # Construct indices to estimate connectivity between the label time course
    # and all source space time courses
    vertices_SD = [src_SD[j]['vertno'] for j in range(2)]
    n_signals_tot = 1 + len(vertices_SD[0]) + len(vertices_SD[1])
    indices = seed_target_indices([0], np.arange(1, n_signals_tot))

    morph_SD = mne.compute_source_morph(src=inv_op_SD['src'],
                                        subject_from=sub_to, subject_to=C.subject_to,
                                        spacing=C.spacing_morph, subjects_dir=C.data_path)
    morph_LD = mne.compute_source_morph(src=inv_op_LD['src'],
                                        subject_from=sub_to, subject_to=C.subject_to,
                                        spacing=C.spacing_morph, subjects_dir=C.data_path)

    for win in np.arange(0, len(C.con_time_window)-1):
        print('[i,win]: ', i, win)

        t_min = C.con_time_window[win]
        t_max = C.con_time_window[win+1]
        stc_F = []
        stc_O = []
        stc_M = []
        stc_LD = []

        for n in np.arange(0, len(stc_f)):
            stc_F.append(stc_f[n].copy().crop(t_min*1e-3, t_max*1e-3))
        for n in np.arange(0, len(stc_o)):
            stc_O.append(stc_o[n].copy().crop(t_min*1e-3, t_max*1e-3))
        for n in np.arange(0, len(stc_m)):
            stc_M.append(stc_m[n].copy().crop(t_min*1e-3, t_max*1e-3))
        for n in np.arange(0, len(stc_ld)):
            stc_LD.append(stc_ld[n].copy().crop(t_min*1e-3, t_max*1e-3))

        for k in np.arange(0, 6):
            print('[i,win,k]: ', i, win, k)
            morphed_labels[k].name = C.rois_labels[k]

            seed_ts_f = mne.extract_label_time_course(stc_F, morphed_labels[k],
                                                      src_SD, mode='mean_flip', return_generator=False)
            seed_ts_o = mne.extract_label_time_course(stc_O, morphed_labels[k],
                                                      src_SD, mode='mean_flip', return_generator=False)
            seed_ts_m = mne.extract_label_time_course(stc_M, morphed_labels[k],
                                                      src_SD, mode='mean_flip', return_generator=False)
            seed_ts_ld = mne.extract_label_time_course(stc_LD, morphed_labels[k],
                                                       src_LD, mode='mean_flip', return_generator=False)

            for f in np.arange(0, len(C.con_freq_band)-1):
                print('[i,win,k,f]: ', i, win, k, f)
                f_min = C.con_freq_band[f]
                f_max = C.con_freq_band[f+1]
                print(f_min, f_max)

                comb_ts_f = zip(seed_ts_f, stc_F)
                comb_ts_o = zip(seed_ts_o, stc_O)
                comb_ts_m = zip(seed_ts_m, stc_M)
                comb_ts_ld = zip(seed_ts_ld, stc_LD)

                con_F, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_f, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

                con_O, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_o, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

                con_M, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_m, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)
                con_LD, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    comb_ts_ld, method=method, mode='fourier', indices=indices,
                    sfreq=500, fmin=f_min, fmax=f_max, faverage=True, n_jobs=10)

                con_SD = (con_F + con_O + con_M)/3

                con_stc_F = mne.SourceEstimate(con_F, vertices=vertices_SD,
                                               tmin=t_min*1e-3, tstep=2e-3, subject=sub_to)
                con_stc_O = mne.SourceEstimate(con_O, vertices=vertices_SD,
                                               tmin=t_min*1e-3, tstep=2e-3, subject=sub_to)
                con_stc_M = mne.SourceEstimate(con_M, vertices=vertices_SD,
                                               tmin=t_min*1e-3, tstep=2e-3, subject=sub_to)
                con_stc_SD = mne.SourceEstimate(con_SD, vertices=vertices_SD,
                                                tmin=t_min*1e-3, tstep=2e-3, subject=sub_to)

                con_stc_LD = mne.SourceEstimate(con_LD, vertices=vertices_SD,
                                                tmin=t_min*1e-3, tstep=2e-3, subject=sub_to)

                stc_total_F[win][k][f] = morph_SD.apply(con_stc_F)
                stc_total_O[win][k][f] = morph_SD.apply(con_stc_O)
                stc_total_M[win][k][f] = morph_SD.apply(con_stc_M)
                stc_total_SD[win][k][f] = morph_SD.apply(con_stc_SD)
                stc_total_LD[win][k][f] = morph_LD.apply(con_stc_LD)

    # with open(stc_F_file_name, "wb") as fp:   #Pickling
    #     pickle.dump(stc_total_F, fp)
    # with open(stc_O_file_name, "wb") as fp:   #Pickling
    #     pickle.dump(stc_total_O, fp)
    # with open(stc_M_file_name, "wb") as fp:   #Pickling
    #     pickle.dump(stc_total_M, fp)
    with open(stc_SD_file_name, "wb") as fp:   #Pickling
        pickle.dump(stc_total_SD, fp)
    # with open(stc_LD_file_name, "wb") as fp:  # Pickling
    #     pickle.dump(stc_total_LD, fp)
    e = time.time()
    print(e-s)


if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(C.subjects))
    # sbj_id = np.array([0, 1])


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]

for s in sbj_ids:
    SN_functional_connectivity_bands_runs(s, method, SN_ROI)
