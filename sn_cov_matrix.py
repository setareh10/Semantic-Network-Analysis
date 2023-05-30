#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:09:34 2020

@author: sr05
"""
import mne
import numpy as np
import os.path as op
import sn_config as C
from mne import compute_covariance
from joblib import Parallel, delayed


# path to raw data
data_path = C.data_path
# Parameters
tmin, tmax = C.tmin_cov, C.tmax_cov

subjects = C.subjects
meth_cov = []
for i in np.arange(0, len(subjects)):


# def SN_cov_matrix(i, data_path, tmin, tmax):
    
    print('Participant : ', i)
    meg = subjects[i]
    print(meg)
    # Complete path to epoched data
    epoch_fname_fruit = data_path + meg + 'block_fruit_epochs-epo.fif'
    epoch_fname_odour = data_path + meg + 'block_odour_epochs-epo.fif'
    epoch_fname_milk = data_path + meg + 'block_milk_epochs-epo.fif'
    epoch_fname_LD = data_path + meg + 'block_LD_epochs-epo.fif'

    # Loading epoched data
    epochs_fruit = mne.read_epochs(epoch_fname_fruit, preload=True)
    epochs_odour = mne.read_epochs(epoch_fname_odour, preload=True)
    epochs_milk = mne.read_epochs(epoch_fname_milk, preload=True)
    epochs_LD = mne.read_epochs(epoch_fname_LD, preload=True)
    epochs_SD = mne.concatenate_epochs([epochs_fruit, epochs_odour,
                                        epochs_milk])

    # Estimating noise covariance matrix from epochs.
    noise_covs_SD = compute_covariance(epochs_SD, tmin=tmin, tmax=tmax,
                                       method='auto', return_estimators=True, rank='info')

    noise_covs_LD = compute_covariance(epochs_LD, tmin=tmin, tmax=tmax,
                                       method='auto', return_estimators=True, rank='info')

    for n in np.arange(len(noise_covs_SD)):
        meth_cov.append([noise_covs_SD[n]['loglik'],
                         noise_covs_SD[n]['method']])
        meth_cov.append([noise_covs_LD[n]['loglik'],
                         noise_covs_LD[n]['method']])
    # return meth_cov
    # Saving covariance matrices

    # cov_fname_SD = data_path + meg + 'block_SD-noise-cov.fif'
    # cov_fname_LD = data_path + meg + 'block_LD-noise-cov.fif'


    # mne.write_cov(cov_fname_SD, noise_covs_SD[0])
    # mne.write_cov(cov_fname_LD, noise_covs_LD[0])
# meth_cov_total = []
# meth_cov_total = Parallel(n_jobs=-1)(delayed(SN_cov_matrix)
#                                      (i, data_path, tmin, tmax)for i in range(len(subjects)))
