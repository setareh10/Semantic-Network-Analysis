#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================
Evaluate leakage among ROIs
=========================================================

PSFs within ROIs are summarised by their first principal component. Leakage
among ROIs is quantified by the inter-correlations of these components, and
visualised as cicular graphs

@author: olaf.hauk@mrc-cbu.cam.ac.uk
Oct 2020
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import (make_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread, get_cross_talk)

from mne.viz import circular_layout, plot_connectivity_circle

import sn_config as C
from SN_semantic_ROIs import SN_semantic_ROIs

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
SN_ROI = SN_semantic_ROIs()

inv_op_name = C.inv_op_name

for i in np.arange(0, 1):
    n_subjects = len(subjects)
    meg = subjects[i]
    mri_subject = MRI_sub[i][1:-1]
    sub_to = MRI_sub[i][1:15]
    print('Participant : ', i)

    # morphing ROIs from fsaverage to each individual
    labels = mne.morph_labels(SN_ROI, subject_to=sub_to,
                              subject_from='fsaverage',
                              subjects_dir=data_path)

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
    n_comp = 5
    stcs_psf, pca_vars = get_point_spread(
        rm, src, labels, mode='pca', n_comp=n_comp, norm=None,
        return_pca_vars=True)
    stcs_ctf, pca_vars = get_cross_talk(
        rm, src, labels, mode='pca', n_comp=n_comp, norm=None,
        return_pca_vars=True)
    

    # We can show the explained variances of principal components per label.
    # Note how they differ across labels, most likely due to their varying
    # spatial extent.

    # label_colors = [(0.06,0.53,0.69,1),(0.02,0.83,0.62,1),\
    #                     (0.02,0.23,0.29,1),(0.93,0.27,0.43,1),\
    #                         (1,0.81,0.4,1),(0.06,0.53,0.69,1)]

    # label_names = ['lATL','rATL','PTC','IFG','AG','PVA']

    label_colors = [label.color for label in labels]
    label_names = [label.name for label in labels]
    # label_names = C.ROIs_lables

    np.set_printoptions(precision=1)
    for [name, var] in zip(label_names, pca_vars):

        print('%s: %.1f%%' % (name, var.sum()))
        print(var)

    # get PSFs from Source Estimate objects into matrix
    psfs_mat = np.zeros([len(labels), rm.shape[0]])

    # Leakage matrix for MNE, get first principal component per label
    for [i, s] in enumerate(stcs_psf):
        psfs_mat[i, :] = s.data[:, 0]

    # Compute label-to-label leakage as Pearson correlation of PSFs
    leakage = np.corrcoef(psfs_mat)

    # Sign of correlation is arbitrary, so take absolute values
    leakage = np.abs(leakage)

    # Plot leakage as circular graph

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(label_names)
    # node_order.extend(lh_labels[::-1])  # reverse the order
    # node_order.extend(rh_labels)

    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 200 strongest connections.
    plt.ion()

    # fig = plt.figure(num=None, figsize=(8, 4), facecolor='black')

    # plot_connectivity_circle(leakage, label_names, n_lines=200,
    #                          node_angles=node_angles, node_colors=label_colors,
    #                          title='Leakage', fig=fig)


###############################################################################
# Plot PSFs for individual labels
# -------------------------------
#
# Just as demonstration, plot first PCA component of first label.



for i in [0,2,3,4,5]:
    idx = [i]
    stc = stcs_ctf[idx[0]]
    max_val = np.max(np.abs(stc.data))
    brain = stc.plot(
        subjects_dir=subjects_dir, subject=mri_subject, hemi='lh', views='lateral',
        clim=dict(kind='value', pos_lims=(0, 0.208 / 5., 0.208)))

    brain.add_text(0.1, 0.9, label_names[idx[0]], 'title', font_size=16)

