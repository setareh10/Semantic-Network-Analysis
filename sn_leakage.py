#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:27:55 2021

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
from surfer import Brain
import matplotlib.pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
# from mne.viz import circular_layout, plot_connectivity_circle
from mne.minimum_norm import (make_inverse_resolution_matrix,
                              get_point_spread, make_inverse_operator,
                              apply_inverse, read_inverse_operator)


# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
subjects_dir = data_path
# Parameters
snr = C.snr
lambda2 = C.lambda2

# EDIT, just pick one ROI
SN_ROI = SN_semantic_ROIs()[0:6]

inv_op_name = C.inv_op_name
leakage_all = np.zeros([6, 6])
leakage_all_norm = np.zeros([6, 6])
stc_all = [0]*6
label_names = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
# label_names = ['lATL', 'rATL', 'PTC', 'rPTC',  'IFG', 'rIFG']

l_n_all = []*18
for i in np.arange(0, len(subjects)):  # np.arange(1, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    mri_subject = MRI_sub[i][1:-1]
    sub_to = MRI_sub[i][1:15]
    print('Participant : ', i)

    # morphing ROIs from fsaverage to each individual
    labels = mne.morph_labels(SN_ROI, subject_to=sub_to,
                              subject_from='fsaverage',
                              subjects_dir=data_path)

    # label_names = [label.name for label in labels]

    for k in np.arange(0, 6):
        # print('[i,win,k]: ',i,win,k)
        labels[k].name = C.rois_labels[k]

    fwd_fname_EMEG = data_path + meg + 'block_EMEG_fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname_EMEG)

    # Convert to fixed source orientations
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True)

    # path to one noise covariance matrix
    cov_fname = data_path + meg + 'block_SD-noise-cov.fif'

    # Loading noise covariance matrix
    cov = mne.read_cov(cov_fname)

    # path to raw data for info
    raw_fname = data_path + meg + 'block_SD_tsss_raw.fif'
    info = mne.io.read_info(raw_fname)

    # make an inverse operator
    inv_op = make_inverse_operator(info, fwd, noise_cov=cov,
                                   fixed=True, loose=0., depth=None,
                                   verbose=None)

    # # Reading inverse operator (from one task)
    # inv_fname = data_path + meg + inv_op_name[0]
    # inv_op = read_inverse_operator(inv_fname)

    # Compute resolution matrix for MNE
    lambda2 = 1. / 3.**2
    method = 'MNE'
    rm = make_inverse_resolution_matrix(fwd, inv_op, method=method,
                                        lambda2=lambda2)

    # Source space used for inverse operator
    src = inv_op['src']

    # Compute first SVD component across PSFs within labels
    # Note the differences in explained variance, probably due to different
    # spatial extents of labels
    
    n_comp =1
    stcs_psf, pca_vars = get_point_spread(
    rm, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
    
    # stcs_psf = get_point_spread(
    # rm, src, labels, mode='mean',  norm=None)
       
    
    stcs_psf = get_point_spread(
        rm, src, labels, mode=None,  norm=None,
        return_pca_vars=False)

    morph = mne.compute_source_morph(src=inv_op['src'], subject_from=stcs_psf[0].subject,
                                     subject_to=C.subject_to, spacing=C.spacing_morph, subjects_dir=C.data_path)

    leakage = np.zeros([6, 6])
    leakage_norm = np.zeros([6, 6])

    # # Restrict the source estimate to our specific labels
    # # the component leakage[r,c] means the total amount of effects in label c,
    # # caused by PSF of label r
    for r in np.arange(0, len(SN_ROI)):
        stc = morph.apply(stcs_psf[r])
        # stc = stcs_psf[r]
        # brain = stc.plot(
        #     subjects_dir=subjects_dir, subject='fsaverage', hemi='lh', views='lateral')
        # brain.add_text(0.1, 0.9, label_names[r], 'title', font_size=16)
        # brain.add_label(SN_ROI[r], borders=True,color='g')

        # stc_all[r]=stc_all[r]+ stc
        for [c, label] in enumerate(SN_ROI):
        # for [c, label] in enumerate(labels):

            stc_label = stc.in_label(label)
            leakage[r, c] = np.mean(np.abs(stc_label.data))
            # leakage[r, c] = np.mean(np.square(stc_label.data))

    l_n_all.append(leakage)
    for r in np.arange(0, len(SN_ROI)):
        leakage_norm[:, r] = leakage[:, r].copy()/leakage[r, r]

    leakage_all = leakage_all + leakage
    leakage_all_norm = leakage_all_norm + leakage_norm

    # leakage_all= leakage_all + leakage


# Plotting the asymmetrical leakage matrix
L= leakage_all_norm
x_pos = np.arange(len(labels))

fig, ax = plt.subplots()
leakage_ave = np.round(L.copy()/len(subjects),4)
# ax.matshow(leakage_ave, cmap=plt.cm.GnBu)
ax.matshow(leakage_ave, cmap=plt.cm.OrRd)


for i in np.arange(len(labels)):
    for j in np.arange(len(labels)):
        c = leakage_ave[j,i]
        ax.text(i, j, str(c), va='center', ha='center')
ax.set_xticks(x_pos)
ax.set_yticks(x_pos)

ax.set_xticklabels(label_names)
ax.set_yticklabels(label_names)
# plt.savefig(C.pictures_path_Source_estimate+ 'leakage_matrix_normalized.png')
# # plt.savefig(C.pictures_path_Source_estimate+ 'leakage_matrix.png')


# # max_val_t=[]
# # for i in np.arange(0,6):
# #     stc = stc_all[i].copy()/18
# #     max_val_t.append(np.max(np.abs(stc.data)))


# for r in [0,2,3,4]:
# # for r in [1]:

#     stc = stc_all[r].copy()/18
#     # max_val=np.max(max_val_t)
#     max_val=np.max(np.abs(stc.data))

#     brain = np.abs(stc).plot(
#         subjects_dir=subjects_dir, subject='fsaverage', hemi='lh', views='lateral',
#         clim=dict(kind='value', lims=(0, max_val / 3., max_val)),colormap='hot')
#     # brain = stc.plot(
#     #     subjects_dir=subjects_dir, subject='fsaverage', hemi='lh', views='lateral')
#     brain.add_text(0.1, 0.9, label_names[r], 'title', font_size=16)
#     brain.add_label(SN_ROI[r], borders=True,color='blue')

# #   # views='coronal'
