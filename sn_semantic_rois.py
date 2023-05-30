#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:15:28 2020

@author: sr05
"""


import numpy as np
import mne
import sn_config as C
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects


def SN_semantic_ROIs():
    # Loading Human Connectom Project parcellation
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=C.data_path, verbose=True)
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',
                                        subjects_dir=data_path)

    ##............................. Control Regions ............................##

    # Temporal area - Splitting STSvp
    label_STSvp = ['L_STSvp_ROI-lh']
    my_STSvp = []
    for j in np.arange(0, len(label_STSvp)):
        my_STSvp.append([label for label in labels if label.name ==
                         label_STSvp[j]][0])

    for m in np.arange(0, len(my_STSvp)):
        if m == 0:
            STSvp = my_STSvp[m]
        else:
            STSvp = STSvp + my_STSvp[m]

    [STSvp1, STSvp2, STSvp3, STSvp4, STSvp5, STSvp6] = mne.split_label(label=STSvp, parts=('L_STSvp1_ROI-lh', 'L_STSvp2_ROI-lh', 'L_STSvp3_ROI-lh', 'L_STSvp4_ROI-lh',
                                                                                           'L_STSvp5_ROI-lh', 'L_STSvp6_ROI-lh',), subject='fsaverage', subjects_dir=C.data_path)

    # Temporal area - Splitting PH
    label_PH = ['L_PH_ROI-lh']
    my_PH = []
    for j in np.arange(0, len(label_PH)):
        my_PH.append(
            [label for label in labels if label.name == label_PH[j]][0])

    for m in np.arange(0, len(my_PH)):
        if m == 0:
            PH = my_PH[m]
        else:
            PH = PH + my_PH[m]

    [PH1, PH2] = mne.split_label(label=PH, parts=(
        'L_PH1_ROI-lh', 'L_PH2_ROI-lh'), subject='fsaverage', subjects_dir=C.data_path)
    [PH21, PH22, PH23, PH24] = mne.split_label(label=PH2, parts=('L_PH21_ROI-lh', 'L_PH22_ROI-lh',
                                                                 'L_PH23_ROI-lh', 'L_PH24_ROI-lh'),
                                               subject='fsaverage', subjects_dir=C.data_path)

    # Temporal area - Splitting TE2p
    label_TE2p = ['L_TE2p_ROI-lh']
    my_TE2p = []
    for j in np.arange(0, len(label_TE2p)):
        my_TE2p.append(
            [label for label in labels if label.name == label_TE2p[j]][0])

    for m in np.arange(0, len(my_TE2p)):
        if m == 0:
            TE2p = my_TE2p[m]
        else:
            TE2p = TE2p + my_TE2p[m]

    [TE2p1, TE2p2] = mne.split_label(label=TE2p, parts=('L_TE2p1_ROI-lh',
                                                        'L_TE2p2_ROI-lh'), subject='fsaverage', subjects_dir=C.data_path)

    # Temporal area
    label_TE1p = ['L_TE1p_ROI-lh']
    my_TE1p = []
    for j in np.arange(0, len(label_TE1p)):
        my_TE1p.append(
            [label for label in labels if label.name == label_TE1p[j]][0])

    for m in np.arange(0, len(my_TE1p)):
        if m == 0:
            TE1p = my_TE1p[m]
        else:
            TE1p = TE1p + my_TE1p[m]

    TG = STSvp1 + STSvp2 + STSvp3 + STSvp4 + TE2p1 + PH24 + TE1p

    ##.......................... Representation Regions .........................##

    # Left ATL area - splitting TE2a
    label_TE2a = ['L_TE2a_ROI-lh']
    my_TE2a = []
    for j in np.arange(0, len(label_TE2a)):
        my_TE2a.append(
            [label for label in labels if label.name == label_TE2a[j]][0])

    for m in np.arange(0, len(my_TE2a)):
        if m == 0:
            l_TE2a = my_TE2a[m]
        else:
            l_TE2a = l_TE2a + my_TE2a[m]

    [l_TE2a1, l_TE2a2, l_TE2a3] = mne.split_label(label=l_TE2a, parts=('L_TE2a1_ROI-lh', 'L_TE2a2_ROI-lh', 'L_TE2a3_ROI-lh'), subject='fsaverage',
                                                  subjects_dir=C.data_path)

    # Left ATL area - splitting TE1m
    label_TE1m = ['L_TE1m_ROI-lh']
    my_TE1m = []
    for j in np.arange(0, len(label_TE1m)):
        my_TE1m.append(
            [label for label in labels if label.name == label_TE1m[j]][0])

    for m in np.arange(0, len(my_TE1m)):
        if m == 0:
            l_TE1m = my_TE1m[m]
        else:
            l_TE1m = l_TE1m + my_TE1m[m]

    [l_TE1m1, l_TE1m2, l_TE1m3] = mne.split_label(label=l_TE1m, parts=('L_TE1m1_ROI-lh', 'L_TE1m2_ROI-lh', 'L_TE1m3_ROI-lh'), subject='fsaverage',
                                                  subjects_dir=C.data_path)
    [l_TE1m11, l_TE1m12, l_TE1m13] = mne.split_label(label=l_TE1m1, parts=('L_TE1m11_ROI-lh', 'L_TE1m12_ROI-lh', 'L_TE1m13_ROI-lh'), subject='fsaverage',
                                                     subjects_dir=C.data_path)
    [l_TE1m21, l_TE1m22, l_TE1m23] = mne.split_label(label=l_TE1m2, parts=('L_TE1m21_ROI-lh', 'L_TE1m22_ROI-lh', 'L_TE1m23_ROI-lh'), subject='fsaverage',
                                                     subjects_dir=C.data_path)

    # Left ATL area
    label_ATL = ['L_TGd_ROI-lh', 'L_TGv_ROI-lh', 'L_TE1a_ROI-lh']

    my_ATL = []
    for j in np.arange(0, len(label_ATL)):
        my_ATL.append(
            [label for label in labels if label.name == label_ATL[j]][0])

    for m in np.arange(0, len(my_ATL)):
        if m == 0:
            l_ATL = my_ATL[m]
        else:
            l_ATL = l_ATL + my_ATL[m]

    l_ATL = l_ATL + l_TE2a2 + l_TE2a3 + l_TE1m13 + l_TE1m23

    # Right ATL area - splitting TE2a
    label_TE2a = ['R_TE2a_ROI-rh']
    my_TE2a = []
    for j in np.arange(0, len(label_TE2a)):
        my_TE2a.append(
            [label for label in labels if label.name == label_TE2a[j]][0])

    for m in np.arange(0, len(my_TE2a)):
        if m == 0:
            r_TE2a = my_TE2a[m]
        else:
            r_TE2a = r_TE2a + my_TE2a[m]

    [r_TE2a1, r_TE2a2, r_TE2a3] = mne.split_label(label=r_TE2a, parts=('R_TE2a1_ROI-rh', 'R_TE2a2_ROI-rh', 'R_TE2a3_ROI-rh'), subject='fsaverage',
                                                  subjects_dir=C.data_path)

    # Right ATL area - splitting TE1m
    label_TE1m = ['R_TE1m_ROI-rh']
    my_TE1m = []
    for j in np.arange(0, len(label_TE1m)):
        my_TE1m.append(
            [label for label in labels if label.name == label_TE1m[j]][0])

    for m in np.arange(0, len(my_TE1m)):
        if m == 0:
            r_TE1m = my_TE1m[m]
        else:
            r_TE1m = r_TE1m + my_TE1m[m]

    [r_TE1m1, r_TE1m2, r_TE1m3] = mne.split_label(label=r_TE1m, parts=('R_TE1m1_ROI-rh', 'R_TE1m2_ROI-rh', 'R_TE1m3_ROI-rh'), subject='fsaverage',
                                                  subjects_dir=C.data_path)

    [r_TE1m31, r_TE1m32, r_TE1m33] = mne.split_label(label=r_TE1m3, parts=('R_TE1m31_ROI-rh', 'R_TE1m32_ROI-rh', 'R_TE1m33_ROI-rh'), subject='fsaverage',
                                                     subjects_dir=C.data_path)

    # Right ATL area
    label_ATL = ['R_TGd_ROI-rh', 'R_TGv_ROI-rh', 'R_TE1a_ROI-rh']

    my_ATL = []
    for j in np.arange(0, len(label_ATL)):
        my_ATL.append(
            [label for label in labels if label.name == label_ATL[j]][0])

    for m in np.arange(0, len(my_ATL)):
        if m == 0:
            r_ATL = my_ATL[m]
        else:
            r_ATL = r_ATL + my_ATL[m]

    r_ATL = r_ATL + r_TE2a2 + r_TE2a3 + r_TE1m33

    ## ............................ Angular Gyrus .............................. ##

    label_AG = ['L_PGi_ROI-lh', 'L_PGp_ROI-lh', 'L_PGs_ROI-lh']

    my_AG = []
    for j in np.arange(0, len(label_AG)):
        my_AG.append(
            [label for label in labels if label.name == label_AG[j]][0])

    for m in np.arange(0, len(my_AG)):
        if m == 0:
            AG = my_AG[m]
        else:
            AG = AG + my_AG[m]

    ## ....................... Inferior Frontal Gyrus  ......................... ##

    label_IFG = ['L_44_ROI-lh', 'L_45_ROI-lh', 'L_47l_ROI-lh', 'L_p47r_ROI-lh']
    my_IFG = []
    for j in np.arange(0, len(label_IFG)):
        my_IFG.append(
            [label for label in labels if label.name == label_IFG[j]][0])

    for m in np.arange(0, len(my_IFG)):
        if m == 0:
            IFG = my_IFG[m]
        else:
            IFG = IFG + my_IFG[m]

    ## ............................, Visual Area ............................... ##
    label_V1 = ['L_V1_ROI-lh', 'L_V2_ROI-lh', 'L_V3_ROI-lh', 'L_V4_ROI-lh']
    # label_V1 = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh','L_V3A_ROI-lh','L_V3B_ROI-lh','L_V6A_ROI-lh','L_V7_ROI-lh','L_V3CD_ROI-lh','L_LO1_ROI-lh','L_LO2_ROI-lh','L_LO3_ROI-lh']

    my_V1 = []
    for j in np.arange(0, len(label_V1)):
        my_V1.append(
            [label for label in labels if label.name == label_V1[j]][0])
    V1 = my_V1[0]
    # for m in np.arange(0,len(my_V1)):
    #     if m==0:
    #         V1 = my_V1[m]
    #     else:
    #         V1 = V1 + my_V1[m]

    return [l_ATL, r_ATL, TG, IFG, AG, V1]
