#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:10:38 2020

@author: sr05
"""
import mne
import numpy as np
import sn_config as C
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_semantic_ROIs_rl import SN_semantic_ROIs_rl

from SN_label_baseline_correction import label_baseline_correction
from scipy import stats as stats
from mne.minimum_norm import apply_inverse, read_inverse_operator
from matplotlib import pyplot as plt
from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc, permutation_cluster_test,
                       f_threshold_mway_rm, f_mway_rm)

# import statsmodels.stats.multicomp as multi

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects = C.subjects
MRI_sub = C.subjects_mri
epochs_names = C.epochs_names
inv_op_name = C.inv_op_name

# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()
# SN_ROI = SN_semantic_ROIs_rl()

X = np.zeros([len(subjects), 2*len(SN_ROI), 1201])
times = np.arange(-300, 901)

for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i][1:15]
    print('Participant : ', i)

    morphed_labels = mne.morph_labels(SN_ROI, subject_to=sub_to,
                                      subject_from='fsaverage', subjects_dir=data_path)
    for n in np.array([0, 1]):
        # Reading epochs
        epo_name = data_path + meg + epochs_names[n]
        epochs = mne.read_epochs(epo_name, preload=True)
        epochs = epochs['words']

        # Reading inverse operator
        inv_fname = data_path + meg + inv_op_name[n]
        inv_op = read_inverse_operator(inv_fname)

        # Evoked responses
        evoked = epochs.average().set_eeg_reference(
            ref_channels='average', projection=True)

        # Applying inverse solution to get sourse signals
        stc = apply_inverse(evoked, inv_op, lambda2, method='MNE',
                            pick_ori=None)
        # stc_corrected = stc_baseline_correction(stc)

        for j in np.arange(0, len(SN_ROI)):
            morphed_labels[j].subject = sub_to

            # label_vertices =  stc_corrected.in_label(morphed_labels[j])
            # X[i,n*len(SN_ROI)+j,:]  = abs(label_vertices.data).copy().mean(0)
            label_vertices = stc.in_label(morphed_labels[j])
            X[i, n*len(SN_ROI)+j, :] = label_baseline_correction(abs(
                label_vertices.data).copy().mean(0), times)


times = epochs.times[0:850]*1e3
X1 = X[:, 0:6, 0:850]
X2 = X[:, 6:12, 0:850]
Y = X1 - X2
Z = []
for j in np.arange(0, 12):
    Z.append(X[:, j, 0:850])
X_SDLD_ATLs = [X1[:, 0, :], X1[:, 1, :], X2[:, 0, :], X2[:, 1, :]]
X_SDLD = [(X1[:, 0, :]+X1[:, 1, :])/2, (X2[:, 0, :]+X2[:, 1, :])/2]
X_ATLs = [(X1[:, 0, :]+X2[:, 0, :])/2, (X1[:, 1, :]+X2[:, 1, :])/2]

# lb=C.ROIs_lables
lb = ['l_ATL', 'r_ATL', 'PTC', 'IFG', 'AG', 'PVA']
T = np.arange(-300, 550)
p_threshold = C.pvalue
n_permutations = 5000
factor_levels = [2, 2]
effects = ['A', 'B', 'A:B']
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
# y_label=['SD/LD' , 'SD/LD by ROIs']
y_label = ['SD/LD', 'ATLs', 'SD/LD by ATLs']
# y_label = ['SD/LD', 'ATLs']

# my_colors = ['mediumblue', 'darkgreen','darkred']
my_colors = ['b', 'g', 'purple']


###############################################################################

# # 2-way repeated measure : F-test and cluster-based correction
# plt.figure(figsize=(9, 9))
# plt.rcParams['font.size'] = '18'

# for e, effect in enumerate(effects):
#     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,
#                                    pvalue=p_threshold)
#     p = 0

#     def stat_fun(*args):
#         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                          effects=effect, return_pvals=False)[0]

#     # The ANOVA returns a tuple f-values and p-values, we will pick the former.
#     tail = 0  # f-test, so tail > 0
#     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#         X_SDLD_ATLs, stat_fun=stat_fun, threshold=f_thresh, tail=tail,
#         n_jobs=4, n_permutations=n_permutations, buffer_size=None, out_type='mask')

#     plt.subplot(len(effects), 1, e+1)
#     # plt.legend(loc='upper left')

#     for i_c, c in enumerate(clusters):
#         c = c[0]

#         if cluster_p_values[i_c] <= 0.05:
#             h = plt.axvspan(times[c.start], times[c.stop - 1],
#                             color='r', alpha=0.3)
#             p = p+1
#             # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')
#         elif 0.05 < cluster_p_values[i_c] <= 0.07:
#             h2 = plt.axvspan(times[c.start], times[c.stop - 1],
#                              color='orange', alpha=0.3)

#         else:
#             h1 = plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
#                              alpha=0.3)
#     # if p == 0:
#     #     plt.legend((h1, ), ('f-test p-value < 0.05', ), loc='upper left')
#     #     plt.legend((h2, ), ('cluster p-value < 0.06', ), loc='upper left')

#     # else:
#     #     plt.legend((h, ), ('cluster p-value < 0.05', ), loc='upper left')

#     hf = plt.plot(times, T_obs, my_colors[e])

#     # if e == 0:
#     # plt.title('Two-Way RM ANOVA and Cluster-Based Permutation Test')
#     if e == 2:
#         plt.xlabel("Time (ms)")
#     plt.ylabel(y_label[e])
#     plt.show()
# plt.savefig(C.pictures_path_Source_estimate + 'two-way_RM_timeseries_ATLs.png')

# # # # plt.close('all')


###############################################################################
# t-test and cluster-based correction for each ROI
# for j in np.arange(0,len(lb)):
for j in np.arange(0,1):


      T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
      Y[:,j,:], n_jobs=4, threshold=t_threshold,
      n_permutations=n_permutations, out_type='mask',tail=0)

      plt.figure()
      plt.subplot(211)
      plt.title('t-test and cluster-based permutation on '+lb[j])
      plt.plot(times, X1[:,j,:].mean(axis=0),'b',label=lb[j]+" time-series: SD")
      plt.plot(times, X2[:,j,:].mean(axis=0),'r',label=lb[j]+" time-series: LD")
      plt.plot(times, Y[:,j,:].mean(axis=0),'m',label=lb[j]+" time-series: SD - LD")

      plt.ylabel("EEG/MEG")
      plt.legend(loc='upper left')
      plt.subplot(212)

      for i_c, c in enumerate(clusters):
          c = c[0]
          if cluster_p_values[i_c] <= 0.05:
              h = plt.axvspan(times[c.start], times[c.stop - 1],
                              color='r', alpha=0.3)
              plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')
          elif 0.05<cluster_p_values[i_c] <= 0.055:
                h2 = plt.axvspan(times[c.start], times[c.stop - 1],
                              color='orange', alpha=0.3)
                plt.legend((h2, ), ('cluster p-value < 0.055', ),loc='upper left')
          else:
              h1= plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                          alpha=0.3)
              plt.legend((h1, ), ('t-test p-value < 0.05', ),loc='upper left')
              # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

      hf = plt.plot(times, T_obs, 'g')

      plt.xlabel("time (ms)")
      plt.ylabel("t-values")
      plt.show()
# plt.savefig(C.pictures_path_Source_estimate+ 't-test_timeseries_'+lb[j]+'.png')

# # plt.close('all')
##############################################################
# t-test and cluster-based correction for each ROI

# max_y=[]
# min_y=[]
# for j in np.arange(0,6):
#     max_y.append(max(np.max(X1[:,j,:].mean(axis=0)),np.max(X2[:,j,:].mean(axis=0))))
#     min_y.append(max(np.min(X1[:,j,:].mean(axis=0)),np.min(X2[:,j,:].mean(axis=0))))


# for j in np.arange(0,6):
#   # for j in np.arange(3,4):
#       T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
#       Y[:,j,:], n_jobs=4, threshold=t_threshold,
#       n_permutations=n_permutations, out_type='mask')

#       fig, ax1 = plt.subplots(figsize=[10,6])

#           # Set general font size
#       plt.rcParams['font.size'] = '18'
#       ax1.plot(times, X1[:, j, :].mean(axis=0), 'navy')
#       ax1.plot(times, X2[:,j,:].mean(axis=0),'teal')
#       ax1.plot(times, Y[:,j,:].mean(axis=0),'purple')
#       ax1.tick_params(axis='y', labelcolor='black')

#       ax2 = ax1.twinx()
#       ax2.plot(times, T_obs.copy(), color='gray', linestyle='dashed')
#       ax2.set_ylabel('t-values', color='gray')  # we already handled the x-label with ax1
#       ax2.tick_params(axis='y', labelcolor='gray')

#       ax1.set_ylim(min(min_y), max(max_y))
#       ax2.set_ylim(-4, 5)


#       for i_c, c in enumerate(clusters):
#           c = c[0]
#           if cluster_p_values[i_c] <= 0.05:
#               h = plt.axvspan(times[c.start], times[c.stop - 1],
#                               color='r', alpha=0.3)
#               # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')
#           elif 0.05<cluster_p_values[i_c] <= 0.075:
#                 h2 = plt.axvspan(times[c.start], times[c.stop - 1],
#                               color='orange', alpha=0.3)

#       ax1.tick_params(axis='x', labelcolor='black')

#       ax1.set_xlabel("Time (ms)",color='black')
#       # ax1.ylabel("EEG/MEG")
#       ax1.set_ylabel("Amplitude",color='black')  # we already handled the x-label with ax1
#       ax2.set_ylabel('t-values',color='gray')  # we already handled the x-label with ax1
#       plt.xscale('linear')
#       plt.show()
#       fig.tight_layout()
#       plt.savefig(C.pictures_path_Source_estimate+ 't-test_results_timeseries_'+lb[j]+'.png')

# # plt.close('all')
##############################################################################
# # t-test and cluster-based correction for interaction of:\
# [SD_lATL,SD_rATL,LD_lATL,LD_rATL] 1 by 1
# ROI_label = ['SD_lATL', 'SD_rATL', 'LD_lATL', 'LD_rATL']
# # times = np.arange(200,400)
# for i in np.arange(0, len(X_SDLD_ATLs)-1):
#     for j in np.arange(i+1, len(X_SDLD_ATLs)):
#         S = X_SDLD_ATLs[i] - X_SDLD_ATLs[j]
#         print(i, j)
#         print(ROI_label[i]+' vs '+ROI_label[j])
#         T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
#             S, n_jobs=4, threshold=t_threshold,
#             n_permutations=n_permutations, out_type='mask')

#         plt.figure()
#         plt.rcParams['font.size'] = '18'

#         plt.subplot(211)
#         plt.title('time-series and cluster-based permutation test')
#         plt.plot(times, X_SDLD_ATLs[i].mean(axis=0), 'b', label=ROI_label[i])
#         plt.plot(times, X_SDLD_ATLs[j].mean(axis=0), 'r', label=ROI_label[j])
#         plt.plot(times, S.mean(axis=0), 'm', label="Contrast")

#         plt.ylabel("EEG/MEG")
#         plt.legend(loc='upper left')
#         plt.subplot(212)

#         for i_c, c in enumerate(clusters):
#             c = c[0]
#             if cluster_p_values[i_c] <= 0.05:
#                 h = plt.axvspan(times[c.start], times[c.stop - 1],
#                                 color='r', alpha=0.3)
#                 plt.legend((h, ), ('cluster p-value < 0.05', ),
#                            loc='upper left')

#             else:
#                 h1 = plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
#                                  alpha=0.3)
#                 # plt.legend((h1, ), ('t-test p-value < 0.05', ),loc='upper left')
#                 # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

#         hf = plt.plot(times, T_obs, 'g')

#         plt.xlabel("Time (ms)")
#         plt.ylabel("t-values")
#         plt.show()
#         plt.savefig(C.pictures_path_Source_estimate +
#                     't-test_timeseries_'+ROI_label[i]+'_'+ROI_label[j]+'.png')

# # plt.close('all')
###############################################################################
# ROI_label = ['SD_lATL', 'SD_rATL', 'LD_lATL', 'LD_rATL']
# my_color = ['mediumblue', 'r', 'teal', 'orange']
# max_y = []
# min_y = []
# for j in np.arange(0, 4):
#     max_y.append(np.max(X_SDLD_ATLs[j]))
#     min_y.append(np.min(X_SDLD_ATLs[j]))


# for i in np.arange(0, len(X_SDLD_ATLs)-1):
#     for j in np.arange(i+1, len(X_SDLD_ATLs)):
#         S = X_SDLD_ATLs[i] - X_SDLD_ATLs[j]
#         T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
#             S, n_jobs=-1, threshold=t_threshold,
#             n_permutations=n_permutations, out_type='mask')

#         fig, ax1 = plt.subplots(figsize=[9, 3])

#         # Set general font size
#         plt.rcParams['font.size'] = '18'
#         ax1.plot(times, X_SDLD_ATLs[i].mean(axis=0), my_color[i])
#         ax1.plot(times, X_SDLD_ATLs[j].mean(axis=0), my_color[j])
#         ax1.plot(times, S.mean(axis=0), 'purple')
#         ax1.tick_params(axis='y', labelcolor='black')

#         ax2 = ax1.twinx()
#         ax2.plot(times, T_obs.copy(), color='gray', linestyle='dashed')
#         # we already handled the x-label with ax1
#         ax2.set_ylabel('t-values', color='gray')
#         ax2.tick_params(axis='y', labelcolor='gray')

#         ax1.set_ylim(min(min_y), max(max_y))
#         ax2.set_ylim(-4, 6)

#         for i_c, c in enumerate(clusters):
#             c = c[0]
#             if cluster_p_values[i_c] <= 0.05:
#                 h = plt.axvspan(times[c.start], times[c.stop - 1],
#                                 color='r', alpha=0.3)
#                 # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')
#             elif 0.05 < cluster_p_values[i_c] <= 0.075:
#                 h2 = plt.axvspan(times[c.start], times[c.stop - 1],
#                                  color='orange', alpha=0.3)

#         ax1.tick_params(axis='x', labelcolor='black')

#         ax1.set_xlabel("Time (ms)", color='black')
#         # ax1.ylabel("EEG/MEG")
#         # we already handled the x-label with ax1
#         ax1.set_ylabel("Amplitude", color='black')
#         # we already handled the x-label with ax1
#         ax2.set_ylabel('t-values', color='gray')
#         plt.xscale('linear')
#         plt.show()
#         fig.tight_layout()
#         plt.savefig(C.pictures_path_Source_estimate +
#                     't-test_timeseries_'+ROI_label[i]+'_'+ROI_label[j]+'.png')
# # # plt.close('all')


# #############################################################################

# # ROI_label = ['SD_lATL','SD_rATL','LD_lATL','LD_rATL']
# # # times = np.arange(200,400)
# # for i in np.arange(0,len(X_SDLD_ATLs)-1):
# #     for j in np.arange(i+1,len(X_SDLD_ATLs)):
# #         S = X_SDLD_ATLs[i] - X_SDLD_ATLs[j]
# #         print(i,j)
# #         print(ROI_label[i]+' vs '+ROI_label[j])
# #         T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
# #         S, n_jobs=4, threshold=t_threshold,
# #         n_permutations=n_permutations, out_type='mask')

# #         plt.figure()
# #         plt.subplot(211)
# #         plt.title('time-series and cluster-based permutation test')
# #         plt.plot(times, X_SDLD_ATLs[i].mean(axis=0),'b',label=ROI_label[i])
# #         plt.plot(times, X_SDLD_ATLs[j].mean(axis=0),'r',label=ROI_label[j])
# #         plt.plot(times, S.mean(axis=0),'m',label="Contrast")

# #         plt.ylabel("EEG/MEG")
# #         plt.legend(loc='upper left')
# #         plt.subplot(212)

# #         for i_c, c in enumerate(clusters):
# #             c = c[0]
# #             if cluster_p_values[i_c] <= 0.05:
# #                 h = plt.axvspan(times[c.start], times[c.stop - 1],
# #                                 color='r', alpha=0.3)
# #                 plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

# #             else:
# #                 h1= plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
# #                             alpha=0.3)
# #                 # plt.legend((h1, ), ('t-test p-value < 0.05', ),loc='upper left')
# #                 # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

# #         hf = plt.plot(times, T_obs, 'g')

# #         plt.xlabel("time (ms)")
# #         plt.ylabel("t-values")
# #         plt.show()
# #         plt.savefig(C.pictures_path_Source_estimate+ 't-test_timeseries_'+ROI_label[i]+'_'+ROI_label[j]+'.png')

# # # # plt.close('all')

# ##############################################################################
# # t-test and cluster-based correction for main effects of: \
# # [SD_lATL,SD_rATL,LD_lATL,LD_rATL]
# # my_label = ['lATL','rATL']
# # my_label = ['SD','LD']

# # # S = X_ATLs[0] - X_ATLs[1]
# # S = X_SDLD[0] - X_SDLD[1]


# # T_obs, clusters, cluster_p_values, h0 = permutation_cluster_1samp_test(
# #     S, n_jobs=4, threshold=t_threshold, connectivity=None,
# #     n_permutations=n_permutations, out_type='mask')

# # plt.figure()
# # plt.subplot(211)
# # plt.title('time-series and cluster-based permutation test')
# # plt.plot(times, X_SDLD[0].mean(axis=0),'b',label=my_label[0])
# # plt.plot(times, X_SDLD[1].mean(axis=0),'r',label=my_label[1])
# # plt.plot(times, S.mean(axis=0),'m',label="Contrast")

# # plt.ylabel("EEG/MEG")
# # plt.legend(loc='upper left')
# # plt.subplot(212)

# # for i_c, c in enumerate(clusters):
# #     c = c[0]
# #     if cluster_p_values[i_c] <= 0.05:
# #         h = plt.axvspan(times[c.start], times[c.stop - 1],
# #                         color='r', alpha=0.3)
# #         plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

# #     else:
# #         h1= plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
# #                     alpha=0.3)
# #         plt.legend((h1, ), ('t-test p-value < 0.05', ),loc='upper left')
# #         # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')

# # hf = plt.plot(times, T_obs, 'g')

# # plt.xlabel("time (ms)")
# # plt.ylabel("t-values")
# # plt.show()
# # plt.savefig(C.pictures_path_Source_estimate+ 't-test_timeseries_'+my_label[0]+'_'+my_label[1]+'.png')

# # # # plt.close('all')
# ###############################################################################
# # Comparison of [SD_lATL,SD_rATL,LD_lATL,LD_rATL] with bar chart
# # X4 = [X1[:,0,:], X1[:,1,:], X2[:,0,:], X2[:,1,:]]
# # means=[]
# # errors=[]
# # for k in np.arange(0,len(X4)):
# #     means.append( X4[k][:,500:700].copy().mean())
# #     errors.append(np.std(X4[k][:,500:700].copy().mean(0)))

# # mean = np.multiply(means,10**12)
# # error = np.multiply(errors,10**12)

# # my_color=['b','r','g','y']

# # labels = ['SD_lATL', 'SD_rATL','LD_lATL', 'LD_rATL']
# # x_pos = np.arange(len(labels))
# # fig, ax = plt.subplots()
# # for i in np.arange(0,4):
# #     ax.bar(x_pos[i], mean[i],
# #             yerr=error[i],
# #             align='center',
# #             alpha=0.5,
# #             ecolor='black',
# #             capsize=10,
# #             color=my_color[i])
# # ax.set_ylabel('Amplitude Mean (X e-12)')
# # ax.set_xticks(x_pos)
# # ax.set_xticklabels(labels)
# # ax.set_title('Comparison of SD/LD in the left and right ATLs')
# # # ax.yaxis.grid(True)
# # plt.tight_layout()
# # plt.show()
# # plt.savefig(C.pictures_path_Source_estimate+ 'bar-chart_SD-LD_ATLs.png')


# # plt.figure(figsize=(8,4))
# # plt.title('Time-series of SD/LD in ATLs')
# # for i in np.arange(0, len(X4)):
# #     plt.plot(times, X4[i].mean(axis=0),my_color[i],label=labels[i])

# # plt.xlabel("time (ms)")
# # plt.ylabel("Amplitudes")
# # plt.legend(loc='upper left')
# # plt.show()
# # h = plt.axvspan(times[500], times[700],color='gray', alpha=0.3)
# # plt.savefig(C.pictures_path_Source_estimate+ 'time-series_SD-LD_ATLs.png')

# # for i in np.arange(0,4):
# #     for j in np.arange(i+1,4):
# #         t,p= stats.ttest_rel(X4[i][:,500:700].copy().mean(1),X4[j][:,500:700].copy().mean(1))
# #         if p<0.05:
# #             print(i,j,': significant')

# ###############################################################
# # X4 = [X1[:,0,:], X1[:,1,:], X2[:,0,:], X2[:,1,:]]
# # means=[]
# # errors=[]
# # for k in np.arange(0,len(X4)):
# #     means.append( X4[k][:,500:700].copy().mean())
# #     errors.append(np.std(X4[k][:,500:700].copy().mean(0)))

# # mean = np.multiply(means,10**12)
# # error = np.multiply(errors,10**12)

# # my_color=['mediumblue','r','teal','orange']

# # labels = ['SD_lATL', 'SD_rATL','LD_lATL', 'LD_rATL']
# # x_pos = np.arange(len(labels))
# # fig, ax = plt.subplots(figsize=(9,5))
# # for i in np.arange(0,4):
# #     ax.bar(x_pos[i], mean[i],
# #             yerr=error[i],
# #             align='center',
# #             alpha=0.4,
# #             ecolor='black',
# #             capsize=10,
# #             color=my_color[i],
# #             width=0.4)
# # ax.set_ylabel('Amplitude Mean (X e-12)')
# # ax.set_xticks(x_pos)
# # ax.set_xticklabels(labels)
# # # ax.set_title('Comparison of SD/LD in the left and right ATLs')
# # # ax.yaxis.grid(True)
# # plt.tight_layout()
# # plt.show()
# # # plt.savefig(C.pictures_path_Source_estimate+ 'bar-chart_SD-LD_ATLs.png')


# # plt.figure(figsize=(10,6))
# # plt.rcParams['font.size'] = '18'

# # # plt.title('Time-series of SD/LD in ATLs')
# # for i in np.arange(0, len(X4)):
# #     plt.plot(times, X4[i].mean(axis=0),my_color[i])

# # plt.xlabel("Time (ms)")
# # plt.ylabel("Amplitudes")
# # plt.legend(loc='upper left')
# # plt.show()
# # h = plt.axvspan(times[500], times[700],color='gray', alpha=0.3)
# # plt.savefig(C.pictures_path_Source_estimate+ 'time-series_SD-LD_ATLs.png')

# for i in np.arange(0, 4):
#     for j in np.arange(i+1, 4):
#         t, p = stats.ttest_rel(X4[i][:, 450:700].copy().mean(
#             1), X4[j][:, 450:700].copy().mean(1))
#         # if p < 0.05:
#         print(i, j, ': significant')
#         print('t, p: ',t, p)
