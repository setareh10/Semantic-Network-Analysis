#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

@author: sr05
"""
import time
import mne
import pickle
import os
import numpy as np
import sn_config as C
from scipy import stats as stats
from matplotlib import pyplot as plt
from SN_semantic_ROIs import SN_semantic_ROIs
from mne.minimum_norm import apply_inverse, read_inverse_operator,\
    source_induced_power
from mne.stats import permutation_cluster_1samp_test, f_threshold_mway_rm,\
    summarize_clusters_stc, permutation_cluster_test,\
    f_mway_rm

start = time.time()
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
X = np.zeros([len(subjects), 2*len(roi), len(freq), 600])
Y = np.zeros([len(subjects), 2*len(roi), len(freq), 600])
n_subjects = len(subjects)

for i in np.arange(0, len(subjects)):

    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    print('Participant : ', i)

    # morphing ROIs from fsaverage to each individual
    morphed_labels = mne.morph_labels(roi, subject_to=sub_to, subject_from='fsaverage',
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
            power, itc = source_induced_power(epochs, inverse_operator=inv_op, freqs=freq, label=morphed_labels[j], lambda2=C.lambda2, method='MNE', baseline=(
                -.300, 0), baseline_mode='percent', n_jobs=-1, n_cycles=n_cycles, zero_mean=True)
            # Averaging across vertices
            # Power
            X[i, n*len(morphed_labels)+j, :, :] = power.copy().mean(0)
            # Phase lock value
            Y[i, n*len(morphed_labels)+j, :, :] = itc.copy().mean(0)

x_pow = np.mean(np.mean(X, 0), 0)
a = 50
b = 200
T = np.arange(-300, 900, 2)
times = T[a:-b]
plt.figure()

plt.rcParams['font.size'] = '14'
plt.imshow(x_pow[:, a:-b],
           extent=[times[1], times[-1], freq[0], freq[-1]],
           aspect='auto', origin='lower', cmap='RdBu_r', vmax=30, vmin=-30)
plt.colorbar()
plt.ylabel('Frequency(Hz)')
plt.xlabel('Time(ms)')


# X_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_power.json'
# Y_file_name=os.path.expanduser('~') +'/my_semnet/source_induced_plv.json'

# # with open(X_file_name, "wb") as fp:   #Pickling
# #     pickle.dump(X, fp)

# # with open(Y_file_name, "wb") as fp:   #Pickling
# #     pickle.dump(Y, fp)


# # end=time.time()
# # print(end-start)
# # common parameteres

# with open(X_file_name, "rb") as fp:   # Unpickling
#     X = pickle.load(fp)

# with open(Y_file_name, "rb") as fp:   # Unpickling
#     Y = pickle.load(fp)
# tail = 0
# a=50
# b=150
# # # T=epochs.times
# T=np.arange(-300,900,2)*1e-3
# t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., n_subjects - 1)
# times=T[a:-b]

# ##############################################################################
# # ## plot each ROI across subjects: columns 0-5->SD, 6:11->LD
# # ## sequence of ROIs: 0:lATL, 1:rATL, 2:TG, 3:IFG, 4:AG, 5:PVA
# for m in np.arange(2,3):
#     plt.figure()
#     plt.subplot(3,1,1)
#     plt.imshow( X[:,m,:,a:-b].copy().mean(0),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r')
#     plt.title('Power of ' + C.rois_labels[m])
#     plt.colorbar()
#     plt.ylabel('SD (HZ)')
#     plt.subplot(3,1,2)
#     plt.imshow( X[:,m+6,:,a:-b].copy().mean(0),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r')
#     plt.colorbar()
#     plt.ylabel('LD (HZ)')

#     plt.subplot(3,1,3)
#     plt.imshow( (X[:,m,:,a:-b].copy().mean(0)-X[:,m+6,:,a:-b].copy().mean(0)),
#                     extent=[times[1], times[-1], freq[0], freq[-1]],
#                     aspect='auto', origin='lower', cmap='RdBu_r')
#     plt.colorbar()
#     plt.ylabel('SD-LD (HZ)')
#     # plt.savefig(C.pictures_path_Source_estimate+ 'TF_zero_mean'+C.ROIs_lables[m]+'.png')
# # plt.close('all')

# ##############################################################################
# ### 2-way repeated measure : F-test and cluster-based correction
# ## Use to compare the SD/LD in all ROIs
# # factor_levels=[2,6]
# # effects=['A','A:B']
# # y_label=['SD-LD','SD-LD by ROIs']
# # # effects=['A']
# # # y_label=['SD-LD']
# # A_all=np.arange(0,12)
# # tail=0
# # S=[]
# # ## A for all ROIs and A_ATL for ATLs
# # for j in A_all:
# #         S.append(X[:,j,:,a:-b])


# # for e , effect in enumerate(effects):

# #     # computing f threshold
# #     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,\
# #                                     pvalue=C.pvalue )
# #     p=0
# #     def stat_fun(*args):
# #         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
# #                           effects=effect, return_pvals=False)[0]

# #     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
# #         S, stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
# #         n_jobs=6, n_permutations=C.n_permutations, buffer_size=None,\
# #         out_type='mask')

# #     T_obs_plot = np.nan * np.ones_like(T_obs)
# #     for c, p_val in zip(clusters, cluster_p_values):
# #         if p_val <= 0.1:
# #             T_obs_plot[c] = T_obs[c]

# #     T_obs_ttest = np.nan * np.ones_like(T_obs)
# #     for r in np.arange(0,X.shape[2]):
# #         for c in np.arange(0,times.shape[0]):
# #             if abs(T_obs[r,c])>f_thresh:
# #                 T_obs_ttest[r,c] =  T_obs[r,c]

# #     vmax = np.max(np.abs(T_obs))
# #     vmin = 0
# #     plt.figure()
# #     # plotting f-values
# #     plt.subplot(311)
# #     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')
# #     plt.title('Power ('+y_label[e]+')')

# #     # Plotting the uncorrected f-test
# #     plt.subplot(312)
# #     plt.imshow(T_obs, cmap=plt.cm.bone,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')

# #     # Plotting the corrected f-test
# #     plt.subplot(313)
# #     plt.imshow(T_obs, cmap=plt.cm.gray,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.xlabel('Time (ms)')
# #     plt.ylabel('Frequency (Hz)')

# #     plt.show()
# #     plt.savefig(C.pictures_path_Source_estimate+ 'two-way_RM_'+y_label[e]+'.png')

# ##############################################################################
# ### t-test and cluster-based correction for each ROI

# # lb=C.ROIs_lables
# # # # difference of SD (0:6) and LD(6:12) for aech ROI and individual
# # Z= X[:,0:6,:,a:-b]-X[:,6:12,:,a:-b]

# # for k in np.arange(0,len(lb)):
# #     T_obs, clusters, cluster_p_values, H0 = \
# #         permutation_cluster_1samp_test(Z[:,k,:,:], n_permutations=C.n_permutations,
# #                                         threshold=t_threshold, tail=tail,
# #                                         connectivity=None,out_type='mask',
# #                                         verbose=True)

# #     T_obs_plot = np.nan * np.ones_like(T_obs)
# #     for c, p_val in zip(clusters, cluster_p_values):
# #         if p_val <= 0.6:
# #             T_obs_plot[c] = T_obs[c]

# #     T_obs_ttest = np.nan * np.ones_like(T_obs)
# #     for r in np.arange(0,X.shape[2]):
# #         for c in np.arange(0,times.shape[0]):
# #             if abs(T_obs[r,c])>t_threshold:
# #                 T_obs_ttest[r,c] =  T_obs[r,c]

# #     vmax = np.max(T_obs)
# #     vmin = np.min(T_obs)
# #     v=max(abs(vmax),abs(vmin))
# #     plt.figure()
# #     # plotting the t-values
# #     plt.subplot(311)
# #     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')
# #     plt.title('Power of '+lb[k])

# #     # plotting the uncorreted t-test
# #     plt.subplot(312)
# #     plt.imshow(T_obs, cmap=plt.cm.bone,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')

# #     # plotting the correted t-test
# #     plt.subplot(313)
# #     plt.imshow(T_obs, cmap=plt.cm.gray,label='cluster-based permutation test',
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,label='cluster-based permutation test',
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.xlabel('Time (ms)')
# #     plt.ylabel('Frequency (Hz)')

# #     plt.show()
#     # plt.savefig(C.pictures_path_Source_estimate+ 't-test_TFR_'+lb[k]+'.png')

# ##############################################################################
# ### 2-way repeated measure : F-test and cluster-based correction
# ## Use to compare the SD/LD in ATLs
# # factor_levels=[2,2]
# # effects=['A','B','A:B']
# # y_label=['SD-LD','ATLs','SD-LD by ATLs']

# # ## indices of ATLs in SD(0,1) and LD(6,7)
# # A_ATL=np.array([0,1,6,7])
# # S=[]
# # for j in A_ATL:
# #     S.append(X[:,j,:,a:-b])

# # for e , effect in enumerate(effects):

# #     # computing f threshold
# #     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,\
# #                                     pvalue= C.pvalue )
# #     p=0
# #     def stat_fun(*args):
# #         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
# #                           effects=effect, return_pvals=False)[0]

# #     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
# #         S, stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
# #         n_jobs=4, n_permutations=C.n_permutations, buffer_size=None,\
# #         out_type='mask')

# #     T_obs_plot = np.nan * np.ones_like(T_obs)
# #     for c, p_val in zip(clusters, cluster_p_values):
# #         if p_val <= C.pvalue:
# #             T_obs_plot[c] = T_obs[c]

# #     T_obs_ttest = np.nan * np.ones_like(T_obs)
# #     for r in np.arange(0,X.shape[2]):
# #         for c in np.arange(0,times.shape[0]):
# #             if abs(T_obs[r,c])>f_thresh:
# #                 T_obs_ttest[r,c] =  T_obs[r,c]

# #     vmax = np.max(np.abs(T_obs))
# #     vmin = 0
# #     plt.figure()
# #     # plotting f-values
# #     plt.subplot(311)
# #     plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')
# #     plt.title('Power ('+y_label[e]+')')

# #     # Plotting the uncorrected f-test
# #     plt.subplot(312)
# #     plt.imshow(T_obs, cmap=plt.cm.bone,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.ylabel('Frequency (Hz)')

# #     # Plotting the corrected f-test
# #     plt.subplot(313)
# #     plt.imshow(T_obs, cmap=plt.cm.gray,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #     plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
# #                 extent=[times[0], times[-1], freq[0], freq[-1]],
# #                 aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #     plt.colorbar()
# #     plt.xlabel('Time (ms)')
# #     plt.ylabel('Frequency (Hz)')

# #     plt.show()
# #     plt.savefig(C.pictures_path_Source_estimate+ 'two-way_RM_ATLs'+y_label[e]+'.png')

# ##############################################################################
# ### t-test and cluster-based correction for SD/LD and ATLs
# # ### :[SD_lATL,SD_rATL,LD_lATL,LD_rATL]
# # ROI_label = ['SD_lATL','SD_rATL','LD_lATL','LD_rATL']
# # X_interaction = [X[:,0,:,50:450], X[:,1,:,50:450],\
# #                  X[:,6,:,50:450:], X[:,11,:,50:450]]

# # for i in np.arange(0,len(X_interaction)-1):
# #     for j in np.arange(i+1,len(X_interaction)):
# #         Z = X_interaction[i] - X_interaction[j]
# #         print(i,j)
# #         print(ROI_label[i]+' vs '+ROI_label[j])
# #         T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
# #         Z, n_jobs=4, threshold=t_threshold, connectivity=None,
# #         n_permutations=C.n_permutations, out_type='mask')


# #         T_obs_plot = np.nan * np.ones_like(T_obs)
# #         for c, p_val in zip(clusters, cluster_p_values):
# #             if p_val <= 0.05:
# #                 T_obs_plot[c] = T_obs[c]

# #         T_obs_ttest = np.nan * np.ones_like(T_obs)
# #         for r in np.arange(0,X.shape[2]):
# #             for c in np.arange(0,times.shape[0]):
# #                 if abs(T_obs[r,c])>t_threshold:
# #                     T_obs_ttest[r,c] =  T_obs[r,c]

# #         vmax = np.max(T_obs)
# #         vmin = np.min(T_obs)
# #         plt.figure()
# #         # plotting t-values
# #         plt.subplot(311)
# #         plt.imshow(T_obs, cmap=plt.cm.RdBu_r,
# #                     extent=[times[0], times[-1], freq[0], freq[-1]],
# #                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax,\
# #                     label=ROI_label[i]+'-'+ROI_label[j])
# #         plt.colorbar()
# #         plt.ylabel('Frequency (Hz)')
# #         plt.title('t-test: '+ ROI_label[i]+'-'+ROI_label[j])
# #         # plt.legend(ROI_label[i]+'-'+ROI_label[j],loc='upper left')

# #         # plotting the uncorrected t-test
# #         plt.subplot(312)
# #         plt.imshow(T_obs, cmap=plt.cm.bone,
# #                     extent=[times[0], times[-1], freq[0], freq[-1]],
# #                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #         plt.imshow(T_obs_ttest, cmap=plt.cm.RdBu_r,
# #                     extent=[times[0], times[-1], freq[0], freq[-1]],
# #                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #         plt.colorbar()
# #         plt.ylabel('Frequency (Hz)')

# #         # plotting the corrected t-test
# #         plt.subplot(313)
# #         plt.imshow(T_obs, cmap=plt.cm.gray,label='cluster-based permutation test',
# #                     extent=[times[0], times[-1], freq[0], freq[-1]],
# #                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

# #         plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,label='cluster-based permutation test',
# #                     extent=[times[0], times[-1], freq[0], freq[-1]],
# #                     aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
# #         plt.colorbar()
# #         plt.xlabel('Time (ms)')
# #         plt.ylabel('Frequency (Hz)')
# #         plt.show()
# # #         plt.savefig(C.pictures_path_Source_estimate+ 't-test_TFR_'+ROI_label[i]+'_'+ROI_label[j]+'.png')
