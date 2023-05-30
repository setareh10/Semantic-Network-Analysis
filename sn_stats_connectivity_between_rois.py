#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:54:56 2020

@author: sr05
"""


import os
import pickle
import mne
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mne.minimum_norm import apply_inverse_epochs, apply_inverse, read_inverse_operator
from mne.connectivity import spectral_connectivity,seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from mne.stats import (permutation_cluster_1samp_test,spatio_temporal_cluster_test,
                       summarize_clusters_stc,permutation_cluster_test, f_threshold_mway_rm,
                       f_mway_rm)
from scipy import stats as stats
from mne.epochs import equalize_epoch_counts
import time

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

method='coh'
# my_stc_coh_SD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
# my_stc_coh_LD=[[[[0]*4 for k in range(6)] for w in range(2)] for i in range(18)]
con_SD=[[]for i in range(18)]
con_LD=[[]for i in range(18)]
con_SD_BL=[[]for i in range(18)]
con_LD_BL=[[]for i in range(18)]
for i in np.arange(0,len(C.subjects)):
    
    # stc_F_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_F_bands_SD_sub'+str(i)+'.json'
    # stc_O_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_O_bands_LD_sub'+str(i)+'.json'
    # stc_M_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'200_M_bands_SD_sub'+str(i)+'.json'
    stc_SD_file_name=os.path.expanduser('~') +'/semnet-project/json_files/connectivity/con_labels_'+method+'200_mean_bands_SD_sub'+str(i)+'.json'
    stc_LD_file_name=os.path.expanduser('~') +'/semnet-project/json_files/connectivity/con_labels_'+method+'200_mean_bands_LD_sub'+str(i)+'.json'
    # stc_SD_bl_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_SD_sub'+str(i)+'.json'
    # stc_LD_bl_file_name=os.path.expanduser('~') +'/my_semnet/json_files/connectivity/con_labels_'+method+'_bl_bands_LD_sub'+str(i)+'.json'

    with open(stc_SD_file_name, "rb") as fp:   # Unpickling
        con_SD[i] = pickle.load(fp)
    
    with open(stc_LD_file_name, "rb") as fp:   # Unpickling
        con_LD[i] = pickle.load(fp)


    # with open(stc_SD_bl_file_name, "rb") as fp:   # Unpickling
    #     con_SD_BL[i] = pickle.load(fp)
    
    # with open(stc_LD_bl_file_name, "rb") as fp:   # Unpickling
    #     con_LD_BL[i] = pickle.load(fp)
#########################################################

# stc_kmax_SD=[]
# stc_kmax_LD=[]

# stc_kmin_SD=[]
# stc_kmin_LD=[]
# w_label=[' 50-250ms',' 250-450ms']
# f_label=['theta','alpha','beta','gamma']

# #########################################################
# label_colors = [(0.06,0.53,0.69,1),(0.06,0.53,0.69,1),(0.02,0.83,0.62,1),\
#                 (0.02,0.23,0.29,1),(0.93,0.27,0.43,1),\
#                 (1,0.81,0.4,1)]   

# # label_names = [label.name for label in SN_ROI]
# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']

# lh_labels = ['lATL','MTG_ITG','IFG','AG','PVA']
# rh_labels = ['rATL']
# node_order = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']
# node_angles = circular_layout(label_names, node_order, start_pos=90,
#                               group_boundaries=[0,1,2,3,4,5])
        

# for w in np.arange(0,2):
#     tmin = C.con_time_window[w]
#     tmax = C.con_time_window[w+1]
#     vmax=[]
#     vmin=[]
#     con_max=[]
#     con_min=[]
#     for f in np.arange(0,4):
#         fmin = C.con_freq_band[f]
#         fmax = C.con_freq_band[f+1] 
#         con_t_SD=0
#         con_t_LD=0
#         for i in np.arange(0,18):                
#             if i==0:
#                 con_t_SD=con_SD[i][w][f]
#                 con_t_LD=con_LD[i][w][f]

#             else:
#                 con_t_SD=con_t_SD+con_SD[i][w][f]
#                 con_t_LD=con_t_LD+con_LD[i][w][f]
        
#         X =  con_t_SD - con_t_LD   
#         X=X/len(C.subjects)      
#         vmax=(X).max()
#         vmin=(X).min()
#         vmax=max(abs(vmax),abs(vmin))
#         fig_con, axes_con = plot_connectivity_circle(
#                             X, label_names, n_lines=None,\
#                             node_angles=node_angles, node_colors=label_colors,\
#                             title='Connectivity_'+method+'_SD-LD, '+\
#                             f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                             f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                             vmin=-vmax, vmax=vmax,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity_'+method+'_SD-LD_'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')
 

#         X =  con_t_SD  
#         X=X/len(C.subjects) 
#         Y =  con_t_LD   
#         Y=Y/len(C.subjects)
#         vmax=max(X.max(),Y.max())
#         vmin=0
#         fig_con, axes_con = plot_connectivity_circle(
#                             X, label_names, n_lines=None,\
#                             node_angles=node_angles, node_colors=label_colors,\
#                             title='Connectivity_'+method+'_SD, '+\
#                             f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                             f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                             vmin=vmin, vmax=vmax)#,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity_'+method+'_SD_'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')
            
#         # X =  con_t_LD   
#         # X=X/len(C.subjects)      
#         # vmax=(X).max()
#         vmin=0
#         fig_con, axes_con = plot_connectivity_circle(
#                             Y, label_names, n_lines=None,\
#                             node_angles=node_angles, node_colors=label_colors,\
#                             title='Connectivity_'+method+'_LD, '+\
#                             f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                             f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                             vmin=vmin, vmax=vmax)#,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity_'+method+'_LD_'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')            
   

# #########################################################
# ## t-test: SD vs LD
label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']

for w in np.arange(0,2):
    tmin = C.con_time_window[w]
    tmax = C.con_time_window[w+1]
    vmax=[]
    vmin=[]
    con_max=[]
    con_min=[]
    for f in np.arange(0,4):
        fmin = C.con_freq_band[f]
        fmax = C.con_freq_band[f+1] 
        
                     
        for m in np.arange(0,6):
            for n in np.arange(m+1,6):
                con_t_SD=np.zeros([18,1])
                con_t_LD=np.zeros([18,1])
                for i in np.arange(0,18):
                # print(m,n)
                # print(label_names[m],'-',label_names[n])       
                                         
                    con_t_SD[i,0]=con_SD[i][w][f][n,m]
                    con_t_LD[i,0]=con_LD[i][w][f][n,m]            
                t_value , p_value = stats.ttest_rel(con_t_SD,con_t_LD)
                if p_value <= C.pvalue/2 and t_value<0:
                    print(w,f,label_names[m],'-',label_names[n],' : significant-LD!', p_value)
                elif p_value <= C.pvalue/2 and t_value>0:
                    print(w,f,label_names[m],'-',label_names[n],' : significant-SD!', p_value)

                    
# #########################################################
## t-test: l vs r in SD
# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']

# for w in np.arange(0,2):
#     tmin = C.con_time_window[w]
#     tmax = C.con_time_window[w+1]
#     vmax=[]
#     vmin=[]
#     con_max=[]
#     con_min=[]
#     for f in np.arange(0,4):
#         fmin = C.con_freq_band[f]
#         fmax = C.con_freq_band[f+1] 
        
                     
#         for m in np.array([2,3,4,5]):
            
#             con_t_l=np.zeros([18,1])
#             con_t_r=np.zeros([18,1])
#             for i in np.arange(0,18):
#             # print(m,n)
#             # print(label_names[m],'-',label_names[n])       
                                     
#                 con_t_l[i,0]=con_SD[i][w][f][m,0]
#                 con_t_r[i,0]=con_SD[i][w][f][m,1]            
#             t_value , p_value = stats.ttest_rel(con_t_l,con_t_r)
#             if p_value <= C.pvalue and t_value<0:
#                 print(w,f,label_names[m],p_value ,'-',' : significant-R!')
#             elif p_value <= C.pvalue and t_value>0:
#                 print(w,f,label_names[m],p_value ,'-',' : significant-L!')
#             # else:
#             #     print(w,f,label_names[m],p_value ,'-',' : not significant!')

                
# ###########################################################
# # # ## t-test: vs BL
# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']


# con_SD0_ave=np.zeros([4,1])
# con_SD1_ave=np.zeros([4,1])
# con_SD_BL_ave=np.zeros([4,1])                    
# for m in np.arange(0,1):
#     for n in np.arange(m+1,2):
#         plt.figure()
#         for f in np.arange(0,4):
#             fmin = C.con_freq_band[f]
#             fmax = C.con_freq_band[f+1] 
            
#             con_t_SD0=np.zeros([18,1])
#             con_t_SD1=np.zeros([18,1])
#             con_t_SD_BL=np.zeros([18,1])
#             for i in np.arange(0,18):                      
#                 con_t_SD0[i,0]=con_SD[i][0][f][n,m]
#                 con_t_SD1[i,0]=con_SD[i][1][f][n,m]
#                 con_t_SD_BL[i,0]=con_SD_BL[i][f][n,m]            
#             con_SD0_ave[f]=con_t_SD0.mean()
#             con_SD1_ave[f]=con_t_SD1.mean()
#             con_SD_BL_ave[f]=con_t_SD_BL.mean()

#         plt.plot([6,12,21,31],con_SD0_ave,'b',label='poststimulus/w1')   
#         plt.plot([6,12,21,31],con_SD1_ave,'g',label='poststimulus/w2')   
#         plt.plot([6,12,21,31],con_SD_BL_ave,'r',label='prestimulus')
#         plt.legend()
#         plt.title('SD Coherence: prestimulus vs poststimulus: '+label_names[m]+'-'+label_names[n])


# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']


# con_SD0_ave=np.zeros([4,1])
# con_SD1_ave=np.zeros([4,1])
# con_SD_BL_ave=np.zeros([4,1])                    
# for m in np.arange(0,1):
#     for n in np.arange(m+1,2):
#         plt.figure()
#         for f in np.arange(0,4):
#             fmin = C.con_freq_band[f]
#             fmax = C.con_freq_band[f+1] 
            
#             con_t_SD0=np.zeros([18,1])
#             con_t_SD1=np.zeros([18,1])
#             con_t_SD_BL=np.zeros([18,1])
#             for i in np.arange(0,18):                      
#                 con_t_SD0[i,0]=con_LD[i][0][f][n,m]
#                 con_t_SD1[i,0]=con_LD[i][1][f][n,m]
#                 con_t_SD_BL[i,0]=con_LD_BL[i][f][n,m]            
#             con_SD0_ave[f]=con_t_SD0.mean()
#             con_SD1_ave[f]=con_t_SD1.mean()
#             con_SD_BL_ave[f]=con_t_SD_BL.mean()

#         plt.plot([6,12,21,31],con_SD0_ave,'b',label='poststimulus/w1')   
#         plt.plot([6,12,21,31],con_SD1_ave,'g',label='poststimulus/w2')   
#         plt.plot([6,12,21,31],con_SD_BL_ave,'r',label='prestimulus')
#         plt.legend()
#         plt.title('LD Coherence: prestimulus vs poststimulus: '+label_names[m]+'-'+label_names[n])

###########################################################
# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']


# con_SD_L_ave=np.zeros([4,1])
# con_SD_R_ave=np.zeros([4,1])
# con_LD_L_ave=np.zeros([4,1]) 
# con_LD_R_ave=np.zeros([4,1])                    
# w=0            
# for n in np.array([2,3,4]):
#     plt.figure()
#     for f in np.arange(0,4):
#         fmin = C.con_freq_band[f]
#         fmax = C.con_freq_band[f+1] 
        
#         con_t_SD_L=np.zeros([18,1])
#         con_t_SD_R=np.zeros([18,1])
#         con_t_LD_L=np.zeros([18,1])
#         con_t_LD_R=np.zeros([18,1])    
#         for i in np.arange(0,18):                      
#             con_t_SD_L[i,0]=con_SD[i][w][f][n,0]
#             con_t_SD_R[i,0]=con_SD[i][w][f][n,1]
#             con_t_LD_L[i,0]=con_LD[i][w][f][n,0]
#             con_t_LD_R[i,0]=con_LD[i][w][f][n,1]  
#         con_SD_L_ave[f]=con_t_SD_L.mean()
#         con_SD_R_ave[f]=con_t_SD_R.mean()
#         con_LD_L_ave[f]=con_t_LD_L.mean()
#         con_LD_R_ave[f]=con_t_LD_R.mean()
#     plt.plot([6,12,21,31],con_SD_L_ave,'b',label='SD/lATL')   
#     plt.plot([6,12,21,31],con_SD_R_ave,'--b',label='SD/rATL')   
#     plt.plot([6,12,21,31],con_LD_L_ave,'r',label='LD/lATL')
#     plt.plot([6,12,21,31],con_LD_R_ave,'--r',label='LD/rATL')

#     plt.legend()
#     plt.title('Coherence: : '+label_names[n])


# ## t-test: vs BL
# label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']


# con_SD0_ave=np.zeros([4,1])
# con_SD1_ave=np.zeros([4,1])
# con_SD_BL_ave=np.zeros([4,1])                    
# for m in np.array([2,3,4,5]):
#         plt.figure()
#         for f in np.arange(0,4):
#             fmin = C.con_freq_band[f]
#             fmax = C.con_freq_band[f+1] 
            
#             con_t_SD0=np.zeros([18,1])
#             con_t_SD1=np.zeros([18,1])
#             con_t_SD_BL=np.zeros([18,1])
#             for i in np.arange(0,18):                      
#                 con_t_SD0[i,0]=con_L[i][0][f][n,0]
#                 con_t_SD1[i,0]=con_R[i][1][f][n,0]
#                 con_t_SD_BL[i,0]=con_SD_BL[i][f][n,m]            
#             con_SD0_ave[f]=con_t_SD0.mean()
#             con_SD1_ave[f]=con_t_SD1.mean()
#             con_SD_BL_ave[f]=con_t_SD_BL.mean()

#         plt.plot([6,12,21,31],con_SD0_ave,'b',label='poststimulus/w1')   
#         plt.plot([6,12,21,31],con_SD1_ave,'g',label='poststimulus/w2')   
#         plt.plot([6,12,21,31],con_SD_BL_ave,'r',label='prestimulus')
#         plt.legend()
#         plt.title('SD Coherence: prestimulus vs poststimulus: '+label_names[m]+'-'+label_names[n])

###########################################################

# ## cluster-based permutation: SD vs LD in each band
# label_colors = [(0.06,0.53,0.69,1),(0.02,0.83,0.62,1),\
#                 (0.02,0.23,0.29,1),(0.93,0.27,0.43,1),\
#                 (1,0.81,0.4,1),(0.06,0.53,0.69,1)]   

# # label_names = [label.name for label in SN_ROI]
# label_names = ['lATL','MTG_ITG','IFG','AG','PVA','rATL']

# lh_labels = ['lATL','MTG_ITG','IFG','AG','PVA']
# rh_labels = ['rATL']
# node_order = ['lATL','MTG_ITG','IFG','AG','PVA','rATL']
# node_angles = circular_layout(label_names, node_order, start_pos=90,
#                               group_boundaries=[0, len(label_names) / 2])
        
# for win in np.arange(0, len(C.con_time_window)-1):
#     tmin = C.con_time_window[win]
#     tmax = C.con_time_window[win+1]
#     for freq in np.arange(0, len(C.con_freq_band)-1):    
        
#         fig_con, axes_con = plot_connectivity_circle(Con_SD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (Coh) SD \n'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                           vmin=-vmax_SD, vmax=vmax_SD,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}_rdbu.jpg')
            
#         fig_con, axes_con = plot_connectivity_circle(Con_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (Coh) LD \n'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                           vmin=-vmax_LD, vmax=vmax_LD,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}_rdbu.jpg')                         
                                
#         fig_con, axes_con = plot_connectivity_circle(con_SD_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (Coh) SD-LD \n'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white',
#                           vmin=-vmax_SD_LD, vmax=vmax_SD_LD,colormap='RdBu')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD-LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}_rdbu.jpg')  
            
            
            
# # # #########################################################            
# # # t_threshold = -stats.distributions.t.ppf(C.pvalue / 2., len(C.subjects) - 1)
# # # not_sig=[]
# # # for w in np.arange(0,2):
# # #     for k in np.arange(0,6):
# # #       # for k in np.array([0,4]):

# # #         for f in np.arange(0,4):
# # #             print('Clustering: ',f_label[f],'/ k:',k, '/ w: ',w)
# # #             X_SD=np.zeros([18,6,6])
# # #             X_LD=np.zeros([18,6,6])
# # #             for i in np.arange(0,18):
# # #                 X_SD[i,:,:]=np.transpose(abs(my_stc_coh_SD[i][w][k][f].data), [1, 0])
# # #                 X_LD[i,:,:]=np.transpose(abs(my_stc_coh_LD[i][w][k][f].data), [1, 0])
            
# # #             Y=X_SD-X_LD
# # #             source_space = mne.grade_to_tris(5)
# # #             connectivity = mne.spatial_tris_connectivity(source_space)
# # #             tstep = my_stc_coh_LD[i][w][k][f].tstep
               
# # #             #     print('Clustering.')
# # #             T_obs, clusters, cluster_p_values, H0 = clu = \
# # #                   spatio_temporal_cluster_1samp_test(Y, connectivity= connectivity,\
# # #                   n_jobs=10,threshold=t_threshold,n_permutations=5000,step_down_p=0.05,t_power=1)
            
# # #             if len(np.where(cluster_p_values<0.05)[0])!=0:           
# # #                 print('significant!')
# # #                 fsave_vertices = [np.arange(10242),np.arange(10242)]            
# # #                 stc_all_cluster_vis = summarize_clusters_stc(clu,tstep=tstep*1000,\
# # #                                       vertices = fsave_vertices)
                 
# # #                 idx = stc_all_cluster_vis.time_as_index(times=stc_all_cluster_vis.times)
# # #                 data = stc_all_cluster_vis.data[:, idx]
# # #                 thresh = max(([abs(data.min()) , abs(data.max())]))
                
# # #                 brain = stc_all_cluster_vis.plot(surface='inflated', hemi='split',subject =\
# # #                     'fsaverage',  subjects_dir=C.data_path, clim=dict(kind='value', pos_lims=\
# # #                     [thresh/10,thresh/5,thresh]),size=(800,400),colormap='mne',time_label=\
# # #                       'SD-LD: '+C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f],views='lateral')
                
# # #                 brain.save_image(C.pictures_path_Source_estimate+'t-test_'+method+'_abs_'+\
# # #                                   C.ROIs_lables[k]+'_'+w_label[w]+'_'+f_label[f]+'.png')
# # #             else:
# # #                 not_sig.append([w,k,f])
           

###############################################################################

## 2-way repeated measure : F-test and cluster-based correction
# plt.figure()
# factor_levels= [2,15]
# effects= ['A', 'B', 'A:B']
# for e , effect in enumerate(effects):
#     f_thresh = f_threshold_mway_rm(len(C.subjects), factor_levels, effects=effect,\
#                                     pvalue= C.p_value )
#     p=0
   
#     def stat_fun(*args):
#         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                           effects=effect, return_pvals=False)[0] 
    
#     # The ANOVA returns a tuple f-values and p-values, we will pick the former.
#     tail = 0  # f-test, so tail > 0
#     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#         X , stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
#         n_jobs=10, n_permutations=n_permutations, buffer_size=None,out_type='mask')
    
       
#     plt.subplot(len(effects),1,e+1)
#     plt.legend(loc='upper left')
        
#     for i_c, c in enumerate(clusters):
#         c = c[0]
        
#         if cluster_p_values[i_c] <= 0.05:
#             h = plt.axvspan(times[c.start], times[c.stop - 1],
#                             color='r', alpha=0.3)
#             p = p+1
#             # plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')
#         elif 0.05<cluster_p_values[i_c] <= 0.055:
#             h2 = plt.axvspan(times[c.start], times[c.stop - 1],
#                             color='orange', alpha=0.3)
           
#         else:
#             h1= plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
#                         alpha=0.3)
#     if p==0:    
#         plt.legend((h1, ), ('f-test p-value < 0.05', ),loc='upper left')
#     else:
#         plt.legend((h, ), ('cluster p-value < 0.05', ),loc='upper left')


#     hf = plt.plot(times, T_obs, my_colors[e])

#     if e==0:
#         plt.title('Two-Way RM ANOVA and Cluster-Based Permutation Test')
#     else:
#           plt.xlabel("time (ms)")
#     plt.ylabel(y_label[e])
#     plt.show()
# plt.savefig(C.pictures_path_Source_estimate+ 'two-way_RM_timeseries_ATLs.png')

# # # plt.close('all')


        