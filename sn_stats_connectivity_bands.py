#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:05:47 2020

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

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

method = 'coh'
my_stc_coh_SD = [[[[0]*4 for k in range(6)]
                  for w in range(2)] for i in range(18)]
my_stc_coh_LD = [[[[0]*4 for k in range(6)]
                  for w in range(2)] for i in range(18)]
# my_stc_coh_SD=[[0]for i in range(18)]
# my_stc_coh_LD=[[0]for i in range(18)]
# my_stc_coh_F=[[[0]*4 for k in range(6)]  for i in range(18)]
# my_stc_coh_O=[[[0]*4 for k in range(6)]  for i in range(18)]
# my_stc_coh_M=[[[0]*4 for k in range(6)]  for i in range(18)]
# my_stc_coh_blSD=[[[0]*4 for k in range(6)]  for i in range(18)]
# my_stc_coh_blLD=[[[0]*4 for k in range(6)]  for i in range(18)]

for i in np.arange(0, len(C.subjects)):

    # stc_F_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'BL_F_bands_sub'+str(i)+'.json'
    # stc_O_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'BL_O_bands_sub'+str(i)+'.json'
    # stc_M_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'BL_M_bands_sub'+str(i)+'.json'
    # stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'BL_SD_bands_sub'+str(i)+'.json'
    # stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'BL_LD_bands_sub'+str(i)+'.json'

    stc_SD_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_mean_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name = os.path.expanduser(
        '~') + '/semnet-project/json_files/connectivity2/stc_'+method+'200_mean_bands_LD_sub'+str(i)+'.json'
    # stc_SD_file_name=os.path.expanduser('~') +'/semnet-project/json_files/connectivity/stc_'+method+'BL_SD_bands_sub'+str(i)+'.json'
    # stc_LD_file_name=os.path.expanduser('~') +'/semnet-project/json_files/connectivity/stc_'+method+'BL_LD_bands_sub'+str(i)+'.json'

    # stc_blLD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_LD_sub'+str(i)+'.json'
    # # stc_F_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_F_bands_SD_sub'+str(i)+'.json'
    # # stc_O_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_O_bands_LD_sub'+str(i)+'.json'
    # # stc_M_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'200_M_bands_SD_sub'+str(i)+'.json'

    # # stc_SD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'_blcorrected_bands_SD_sub'+str(i)+'.json'
    # # stc_LD_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/stc_'+method+'_blcorrected_bands_LD_sub'+str(i)+'.json'
    # stc_F_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_F_sub'+str(i)+'.json'
    # stc_M_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_M_sub'+str(i)+'.json'
    # stc_O_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_O_sub'+str(i)+'.json'

    with open(stc_SD_file_name, "rb") as fp:   # Unpickling
        my_stc_coh_SD[i] = pickle.load(fp)

    with open(stc_LD_file_name, "rb") as fp:   # Unpickling
        my_stc_coh_LD[i] = pickle.load(fp)

    # with open(stc_F_file_name, "rb") as fp:   # Unpickling
    #     my_stc_coh_F[i] = pickle.load(fp)

    # with open(stc_O_file_name, "rb") as fp:   # Unpickling
    #     my_stc_coh_O[i] = pickle.load(fp)

    # with open(stc_M_file_name, "rb") as fp:   # Unpickling
    #     my_stc_coh_M[i] = pickle.load(fp)

    # with open( stc_SD_file_name, "rb") as fp:   # Unpickling
    #     my_stc_coh_blSD[i] = pickle.load(fp)

    # with open( stc_LD_file_name, "rb") as fp:   # Unpickling
    #     my_stc_coh_blLD[i] = pickle.load(fp)
#########################################################

stc_kmax_SD = []
stc_kmax_LD = []

stc_kmin_SD = []
stc_kmin_LD = []
w_label = [' 50-250ms', ' 250-450ms']
f_label = ['theta', 'alpha', 'beta', 'gamma']

#########################################################
# Plot average coh maps for two windows, four frequency
# bands, and two conditions
# for w in np.arange(0,1):
#     vmax=[]
#     vmin=[]
#     for k in np.arange(0,6):
#         stc_max=[]
#         stc_min=[]
#         for f in np.arange(0,4):
#             stc_t_SD=0
#             stc_t_LD=0
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=my_stc_coh_LD[i][w][k][f]

#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][w][k][f]

#             stc_t =  stc_t_SD - stc_t_LD
#             stc_max.append((stc_t/len(C.subjects)).data.max())
#             stc_min.append((stc_t/len(C.subjects)).data.min())

#         vmax.append(max(stc_max))
#         vmin.append(min(stc_min))

#     for k in np.arange(0,1):
#         for f in np.arange(1,2):
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=my_stc_coh_LD[i][w][k][f]

#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][w][k][f]

#             stc_t=(stc_t_SD - stc_t_LD)/len(C.subjects)

#             v_max=vmax[k]
#             v_min=vmin[k]
#             v_max=max(abs(v_max),abs(v_min))


#             print('Coherence_SD : '+C.rois_labels[k]+'-'+w_label[w])
#             brain = stc_t.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD-LD: '+C.rois_labels[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='value', pos_lims=(v_max/6,v_max/3,v_max)),colormap='mne')

#             brain = stc_t.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD-LD: '+C.rois_labels[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='percent', pos_lims=(50,75,100)),colormap='mne')

#             brain.save_image(C.pictures_path_Source_estimate+method+'_alltrials_SD-LD_'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')

#########################################################
# Plot average coh maps for two windows, four frequency
# bands, and two conditions
# for w in np.arange(0,1):
#     vmax=[]
#     vmin=[]
#     for k in np.arange(0,6):
#         stc_max_SD=[]
#         stc_max_LD=[]
#         stc_min_SD=[]
#         stc_min_LD=[]
#         for f in np.arange(0,4):
#             stc_t_SD=0
#             stc_t_LD=0
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=my_stc_coh_LD[i][w][k][f]

#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][w][k][f]

#             stc_max_SD.append((stc_t_SD/len(C.subjects)).data.max())
#             stc_max_LD.append((stc_t_LD/len(C.subjects)).data.max())
#             stc_min_SD.append((stc_t_SD/len(C.subjects)).data.min())
#             stc_min_LD.append((stc_t_LD/len(C.subjects)).data.min())
#         vmax.append(max(max(stc_max_SD),max(stc_max_LD)))
#         vmin.append(min(min(stc_min_SD),min(stc_min_LD)))

#     for k in np.arange(0,1):
#         for f in np.arange(1,2):
#             for i in np.arange(0,18):
#                 if i==0:
#                     stc_t_SD=my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=my_stc_coh_LD[i][w][k][f]

#                 else:
#                     stc_t_SD=stc_t_SD+my_stc_coh_SD[i][w][k][f]
#                     stc_t_LD=stc_t_LD+my_stc_coh_LD[i][w][k][f]

#             stc_T_SD=stc_t_SD/len(C.subjects)
#             stc_T_LD=stc_t_LD/len(C.subjects)

#             v_max=vmax[k]
#             v_min=vmin[k]
#             # vmid=(v_max+v_min)/4

#             v_max=max(abs(v_max),abs(v_min))
#             v_min=v_max/10
#             vmid=v_max/5

#             print('Coherence_SD : '+C.ROIs_lables[k]+'-'+w_label[w])
#             brain = stc_T_SD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_SD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='value', pos_lims=(v_min,vmid,v_max)))
#             brain.save_image(C.pictures_path_Source_estimate+method+'_alltrails_SD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')

#             print('Coherence_LD : '+C.ROIs_lables[k]+'-'+w_label[w]+'-'+f_label[f])

#             brain = stc_T_LD.plot(surface='inflated', hemi='split',
#                       time_label=method+'_LD : '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],
#                       subjects_dir=C.data_path,size=([800,400]),
#                       clim=dict(kind='value', pos_lims=(v_min,vmid,v_max)))
#             brain.save_image(C.pictures_path_Source_estimate+method+'_alltrails_LD-'+C.ROIs_lables[k]+'_'+w_label[w][:-3]+'_'+f_label[f]+'.png')

#########################################################
# # ## cluster-based permutation: SD vs LD in each band
# p=0.05
# t_threshold = -stats.distributions.t.ppf(p / 2., len(C.subjects) - 1)
# not_sig=[]
# sig=[]
# for w in np.arange(0,1):
#     for k in np.arange(0,1):
#       # for k in np.array([0,4]):

#         for f in np.arange(0,1):
#             print('Clustering: ',f_label[f],'/ k:',k, '/ w: ',w)
#             # X_M=np.zeros([18,1,20484])
#             # X_O=np.zeros([18,1,20484])
#             # X_F=np.zeros([18,1,20484])
#             X_SD=np.zeros([18,1,20484])
#             X_LD=np.zeros([18,1,20484])

#             # bl_SD=np.zeros([18,1,20484])
#             # bl_LD=np.zeros([18,1,20484])
#             for i in np.arange(0,18):
#                 # X_M[i,:,:]=np.transpose(abs(my_stc_coh_M[i][w][k][f].data), [1, 0])
#                 # X_O[i,:,:]=np.transpose(abs(my_stc_coh_O[i][w][k][f].data), [1, 0])
#                 # X_F[i,:,:]=np.transpose(abs(my_stc_coh_F[i][w][k][f].data), [1, 0])
#                 X_SD[i,:,:]=np.transpose(abs(my_stc_coh_SD[i][w][k][f].data), [1, 0])
#                 X_LD[i,:,:]=np.transpose(abs(my_stc_coh_LD[i][w][k][f].data), [1, 0])
#                 # X_M[i,:,:]=np.transpose(abs(my_stc_coh_M[i][k][f].data), [1, 0])
#                 # X_O[i,:,:]=np.transpose(abs(my_stc_coh_O[i][k][f].data), [1, 0])
#                 # X_F[i,:,:]=np.transpose(abs(my_stc_coh_F[i][k][f].data), [1, 0])
#                 # bl_SD[i,:,:]=np.transpose(abs(my_stc_coh_blSD[i][k][f].data), [1, 0])
#                 # bl_LD[i,:,:]=np.transpose(abs(my_stc_coh_blLD[i][k][f].data), [1, 0])
#             Y =  X_SD - X_LD
#             # Y =  X_F - bl_LD

#             # Y = (X_F+X_O+X_M)/3-(X_LD)

#             source_space = mne.grade_to_tris(5)
#             connectivity = mne.spatial_tris_connectivity(source_space)
#             connectivity = None

#             tstep = my_stc_coh_SD[i][k][f].tstep

#             #     print('Clustering.')
#             # T_obs, clusters, cluster_p_values, H0 = clu = \
#             #       spatio_temporal_cluster_1samp_test(Y, connectivity= connectivity,\
#             #       n_jobs=10,threshold=t_threshold,n_permutations=5000,step_down_p=0.05,t_power=1)
#             T_obs, clusters, cluster_p_values, H0 = clu = \
#                   mne.stats.permutation_cluster_1samp_test(Y, \
#                   n_jobs=-1,threshold=t_threshold,n_permutations=5000,step_down_p=0.05,t_power=1)


#             if len(np.where(cluster_p_values<p)[0])!=0:
#                 print('significant!')
#                 sig.append([w,k,f])
#                 # fsave_vertices = [np.arange(10242),np.arange(10242)]
#                 # stc_all_cluster_vis = summarize_clusters_stc(clu,tstep=tstep*1000,\
#                 #                       vertices = fsave_vertices)

#                 # idx = stc_all_cluster_vis.time_as_index(times=stc_all_cluster_vis.times)
#                 # data = stc_all_cluster_vis.data[:, idx]
#                 # thresh = max(([abs(data.min()) , abs(data.max())]))

#                 # brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split', subject =\
#                 #     'fsaverage', subjects_dir=C.data_path, clim=dict(kind='value', pos_lims=\
#                 #     [thresh/4,thresh/2,thresh]), size=(800,400), colormap='mne', time_label=\
#                 #       'BL (F-LD): ' + C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f], views='lateral')


#                 # # brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split',subject =\
#                 # #     'fsaverage',  subjects_dir=C.data_path,size=(800,400),colormap='mne',time_label=\
#                 # #       'SD-LD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],views='lateral')

#                 # brain.save_image(C.pictures_path_Source_estimate +'t-test_' + method +'_BL(F-LD)_' + str(p) +'_' + \
#                 #                  C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f] + '.png')


#             else:
#                 not_sig.append([w,k,f])


##############################################################################
# # ## cluster-based permutation: SD vs LD in each band
p = 0.05
t_threshold = -stats.distributions.t.ppf(p / 2., len(C.subjects) - 1)
not_sig = []
sig = []
vertices = [np.arange(10242),np.arange(10242)]

for w in np.arange(0, 2):
    for k in np.arange(0, 6):
      # for k in np.array([0,4]):

        for f in np.arange(0, 4):
            print('Clustering: ', f_label[f], '/ k:', k, '/ w: ', w)

            X_SD = np.zeros([18, 20484])
            X_LD = np.zeros([18, 20484])

            for i in np.arange(0, 18):

                X_SD[i, :] = (abs(my_stc_coh_SD[i][w][k][f].data)).reshape(20484)
                X_LD[i, :] = (abs(my_stc_coh_LD[i][w][k][f].data)).reshape(20484)
                # X_SD[i, :] = (abs(my_stc_coh_SD[i][k][f].data)).reshape(20484)
                # X_LD[i, :] = (abs(my_stc_coh_LD[i][k][f].data)).reshape(20484)

            Y = X_SD - X_LD

            source_space = mne.grade_to_tris(5)
            # adjacency = mne.spatial_tris_connectivity(source_space)
            # adjacency = None

            tstep = my_stc_coh_SD[i][w][k][f].tstep
            # tstep = my_stc_coh_SD[i][k][f].tstep

            # T_obs, clusters, cluster_p_values, H0 = clu = \
            #     mne.stats.permutation_cluster_1samp_test(Y, connectivity=adjacency,
            #                                              n_jobs=-1, 
            #                                              threshold=t_threshold,
            #                                              n_permutations=5000, 
            #                                              step_down_p=0.05, 
            #                                              t_power=1, tail=0)
            T_obs, clusters, cluster_p_values, H0 = clu = \
                mne.stats.permutation_cluster_1samp_test(Y, 
                                                         n_jobs=-1, 
                                                         threshold=t_threshold,
                                                         n_permutations=5000, 
                                                         step_down_p=0.05, 
                                                         t_power=1, tail=0)
            if len(np.where(cluster_p_values < p)[0]) != 0:
                print('significant!','p-value: ',cluster_p_values)
                # sig.append([w, k, f ])
                T_obs_plot = np.nan * np.ones_like(T_obs)
                for c, p_val in zip(clusters, cluster_p_values):
                    if p_val <= 0.025:
                        T_obs_plot[c] = T_obs[c]
                        sig.append([w, k, f, p_val ])    
                stc_T_obs = mne.SourceEstimate(T_obs, vertices=vertices,
                                               tmin=50*1e-3, tstep=2e-3,
                                               subject='fsaverage')
                max_t=stc_T_obs.data.max()
                brain = stc_T_obs.plot(surface='inflated', hemi='split',subject =\
                    'fsaverage',  subjects_dir=C.data_path,size=(800,400),colormap='mne',time_label=\
                      'SD-LD: '+C.rois_labels[k]+'_'+w_label[w]+'_'+f_label[f],views='lateral',
                      clim=dict(kind='value', pos_lims=[t_threshold,(max_t+t_threshold)/2,max_t]))

                # brain = stc_T_obs.plot(surface='inflated', hemi='split',subject =\
                #     'fsaverage',  subjects_dir=C.data_path,size=(800,400),colormap='mne',time_label=\
                #       'SD-LD: '+C.rois_labels[k]+'_'+w_label[w]+'_'+f_label[f],views='lateral')

                # brain.save_image(C.pictures_path_Source_estimate +'t-test_' + method +'_SD-LD_' + str(p) +'_' + \
                #                   C.rois_labels[k] + '_' + w_label[w] + '_' + f_label[f] + '.png')

            else:
                not_sig.append([w, k, f])

# plt.close('all')