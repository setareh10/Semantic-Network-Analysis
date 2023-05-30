#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:57:34 2021

@author: sr05
"""

"""
=========================================================
Compute power and phase lock value in 6 ROIs
=========================================================

Compute time-frequency maps of power and phase lock in the
source space.
The inverse method is linear based on MNE inverse operator.

There are 4 sections needed to be uncomment based on the
purpose:1) 2-way RM to compare SD/LD and interaction by ROIs.
2) t-test on each ROI to compare SD/LD. 3) 2-way RM to compare
SD/LD and lATL/rATL and the their interaction. 4) t-test on any
combination of: SD_lATL, SD_rATL, LD_lATL, LD_rATL

"""
import mne
import numpy as np
import sn_config as C
from matplotlib import pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse, read_inverse_operator, \
    source_induced_power
from joblib import Parallel, delayed

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri_files
# MRI_sub = C.subjects_mri

# Parameters
snr = C.snr
lambda2 = C.lambda2
roi = SN_semantic_ROIs()
freq = np.arange(6, 40, 2)  # define frequencies of interest
n_cycles = freq / 3
epochs_names = C.epochs_names
inv_op_name = C.inv_op_name
Y = []
n_subjects = len(subjects)


def tfr_roi(i, subjects, MRI_sub, roi, freq, n_cycles):
    X = np.zeros([2 * len(roi), len(freq), 600])
    meg = subjects[i]
    sub_to = MRI_sub[i][1: 15]
    print('Participant : ', i)

    # morphing ROIs from fsaverage to each individual
    morphed_labels = mne.morph_labels(roi, subject_to=sub_to,
                                      subject_from='fsaverage',
                                      subjects_dir=data_path)
    # Reading epochs for SD(n=0)/LD(n=1)
    for n in np.array([0, 1]):
        epo_name = data_path + meg + epochs_names[n]
        epochs = mne.read_epochs(epo_name, preload=True)
        epochs = epochs['words'].resample(500)

        # Reading inverse operator
        inv_fname = data_path + meg + inv_op_name[n]
        inv_op = read_inverse_operator(inv_fname)

        # Computing the power and phase lock value for each ROI
        for j in np.arange(0, len(morphed_labels)):
            print('Participant: ', i, '/ condition: ', n, '/ ROI: ', j)
            power, itc = source_induced_power(epochs, inverse_operator=inv_op,
                                              freqs=freq,
                                              label=morphed_labels[j],
                                              lambda2=C.lambda2, method='MNE',
                                              baseline=(
                                                  -.300, 0),
                                              baseline_mode='percent',
                                              n_jobs=-1, n_cycles=n_cycles,
                                              zero_mean=True)
            # Averaging across vertices
            # Power
            X[i, n * len(morphed_labels) + j, :, :] = power.copy().mean(0)
    return X


Y = Parallel(n_jobs=-1)(delayed(tfr_roi)
                        (i, subjects, MRI_sub, roi, freq, n_cycles)
                        for i in np.arange(len(subjects)-16))
