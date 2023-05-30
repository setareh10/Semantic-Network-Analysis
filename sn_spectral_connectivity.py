#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:12:43 2020

@author: sr05
"""

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
from SN_semantic_ROIs import SN_semantic_ROIs
from SN_stc_baseline_correction import stc_baseline_correction
from SN_matrix_mirror import matrix_mirror 
from mne.stats import (permutation_cluster_1samp_test,
                       summarize_clusters_stc,permutation_cluster_test)
from scipy import stats as stats
# from mne.epochs import equalize_epoch_counts
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm, summarize_clusters_stc)
from matplotlib import pyplot as plt
import statsmodels.stats.multicomp as multi
from SN_label_baseline_correction import label_baseline_correction

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri
# Parameters
snr = C.snr
lambda2 = C.lambda2
label_path = C.label_path
SN_ROI = SN_semantic_ROIs()    
X_SD = np.zeros([len(subjects),len(SN_ROI),250])
X_LD = np.zeros([len(subjects),len(SN_ROI),250])
times = np.arange(-300,901)



for win in np.arange(0, len(C.con_time_window)):
    tmin = C.con_time_window[win]
    tmax = C.con_time_window[win]+ C.con_time_window_len
    for freq in np.arange(0, len(C.con_freq_band)-1):
        fmin = C.con_freq_band[freq]
        fmax = C.con_freq_band[freq+1]     
        for i in np.arange(0, len(subjects)):
            n_subjects = len(subjects)
            meg = subjects[i]
            sub_to = MRI_sub[i][1:15]
        
            # print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
            print('Participant : ' , i)
        
            morphed_labels = mne.morph_labels(SN_ROI,subject_to=data_path+sub_to,\
                          subject_from='fsaverage',subjects_dir=data_path)
                
        
            # Reading epochs
            epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
            epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
                
            epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
            epochs_ld = mne.read_epochs(epo_name_LD, preload=True)
        
            
            epochs_SD = epochs_sd['words']
            epochs_LD = epochs_ld['words']
        
            # Equalize trial counts to eliminate bias (which would otherwise be
            # introduced by the abs() performed below)
            equalize_epoch_counts([epochs_SD, epochs_LD])
            
            # Reading inverse operator
            inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
            inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'
        
            inv_op_SD = read_inverse_operator(inv_fname_SD) 
            inv_op_LD = read_inverse_operator(inv_fname_LD) 
                        
            stc_sd = apply_inverse_epochs(epochs_SD, inv_op_SD,lambda2,method ='MNE', 
                                  pick_ori="normal", return_generator=False)
            stc_ld = apply_inverse_epochs(epochs_LD, inv_op_LD,lambda2,method ='MNE',
                                    pick_ori="normal", return_generator=False)
            times=epochs_sd.times
            stc_SD = [stc_baseline_correction(stc,times) for stc in stc_sd]
            stc_LD = [stc_baseline_correction(stc,times) for stc in stc_ld]
        
            src_SD = inv_op_SD['src']
            src_LD = inv_op_LD['src']
            for k in np.arange(0,6):
                    morphed_labels[k].subject = sub_to  
                    
            label_sd = mne.extract_label_time_course(stc_SD, morphed_labels, \
                       src_SD, mode='mean_flip',return_generator=False)
            label_ld = mne.extract_label_time_course(stc_LD, morphed_labels, \
                       src_LD, mode='mean_flip',return_generator=False)
            
            label_SD=np.zeros([len(label_sd),6,250])
            label_LD=np.zeros([len(label_ld),6,250])
        
            for m in np.arange(0,len(label_sd)):
                label_SD[m,:,:]=label_sd[m][:,tmin:tmax]
                label_LD[m,:,:]=label_ld[m][:,tmin:tmax]
        
            # label_vertices=[]
            # SD=[]
            # LD=[]
            # for n in np.arange(0,len(stc_sd)):
            #     print('Participant : ' , i, '/ win : ',win, '/ freq : ',freq)
        
            #     c=np.zeros([6,250])
            #     d=np.zeros([6,250])
            #     for k in np.arange(0,6):
            #         morphed_labels[k].subject = sub_to  
            #         a_sd=stc_sd[n].in_label(morphed_labels[k])
            #         a_ld=stc_ld[n].in_label(morphed_labels[k])
           
            #         c[k,:]=label_baseline_correction(abs(a_sd.copy().data).mean(0),times)[tmin:tmax,]
            #         d[k,:]=label_baseline_correction(abs(a_ld.copy().data).mean(0),times)[tmin:tmax,]
            #     SD.append(c)
            #     LD.append(d)
        
            con_SD,freqs_SD,times_SD,n_epochs_SD,n_tapers_SD=spectral_connectivity(\
                        label_SD, method='wpli2_debiased', mode='fourier',\
                        sfreq=C.sfreq, n_jobs=6)   
            
            con_LD,freqs_LD,times_LD,n_epochs_LD,n_tapers_LD=spectral_connectivity(\
                        label_LD, method='wpli2_debiased', mode='fourier',\
                        sfreq=C.sfreq, n_jobs=6)



            C.im_coh_sd[i, freq, win, :, :] = con_SD.copy().mean(2)
            C.im_coh_ld[i, freq, win, :, :] = con_LD.copy().mean(2)

# C.ImCoh_SD_LD= C.ImCoh_SD.copy().mean(2)- C.ImCoh_LD.copy().mean(2)
# C.ImCoh_SD_sorted = C.ImCoh_SD.copy().mean(2)  
# C.ImCoh_LD_sorted = C.ImCoh_LD.copy().mean(2)  


# X_SD = matrix_mirror(np.swapaxes(C.ImCoh_SD.copy(),2,0))
# X_LD = matrix_mirror(np.swapaxes(C.ImCoh_LD.copy(),2,0))

# X_sd =np.swapaxes(C.ImCoh_SD.copy(),2,0)
# X_ld = np.swapaxes(C.ImCoh_LD.copy(),2,0)
X_SD=np.zeros([18,3,2,15])
X_LD=np.zeros([18,3,2,15])

c=0
for i in np.arange(0,6):
    for j in np.arange(i+1,6):
        # print(c)
        X_SD[:,:,:,c]= C.im_coh_sd[:, :, :, j, i]
        X_LD[:,:,:,c]= C.im_coh_ld[:, :, :, j, i]
        c=c+1

# X = np.zeros([18,12,15])
# X[:,0,:] = X_SD[:,0,0,:]
# X[:,1,:] = X_SD[:,0,1,:]
# X[:,2,:] = X_SD[:,1,0,:]
# X[:,3,:] = X_SD[:,1,1,:]
# X[:,4,:] = X_SD[:,2,0,:]
# X[:,5,:] = X_SD[:,2,1,:]
# X[:,6,:] = X_LD[:,0,0,:]
# X[:,7,:] = X_LD[:,0,1,:]
# X[:,8,:] = X_LD[:,1,0,:]
# X[:,9,:] = X_LD[:,1,1,:]
# X[:,10,:] = X_LD[:,2,0,:]
# X[:,11,:] = X_LD[:,2,1,:]
# effects=['B']

# fval, pval=f_mway_rm(X, factor_levels=factor_levels,\
#                       effects=effects,correction=False)




# S=[]
# for i in np.arange(0,3):
#     for j in np.arange(0,2):
#         S.append(X_SD[:,i,j,:])
        
# for i in np.arange(0,3):
#     for j in np.arange(0,2):
#         S.append(X_LD[:,i,j,:])        

# factor_levels=[2,3,2]
# effects=['C']
# p_threshold=0.05
# n_permutations=5000
# for e , effect in enumerate(effects):
#     f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects=effect,\
#                                     pvalue= p_threshold )
#     p=0
   
#     def stat_fun(*args):
#         return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
#                           effects=effect, return_pvals=False)[0] 
    
#     # The ANOVA returns a tuple f-values and p-values, we will pick the former.
#     tail = 0  # f-test, so tail > 0
#     T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
#         S , stat_fun=stat_fun, threshold=f_thresh, tail=tail,\
#         n_jobs=4, n_permutations=n_permutations, buffer_size=None,out_type='mask')
    



# effect_labels = ['SD/LD', 'Freq', 'SD/LD by freq','time','SD/LD by time',\
#                   'freq by time','all']

# # let's visualize our effects by computing f-images
# for effect, sig, effect_label in zip(fvalss, pvalss, effect_labels):
#     plt.figure()
#     # show naive F-values in gray
#     plt.imshow(effect.reshape(6, 6), cmap=plt.cm.RdBu_r, extent=[1,
#                 6, 1, 6], aspect='auto',
#                 origin='lower')
#     # create mask for significant Time-frequency locations
#     # effect = np.ma.masked_array(effect, [sig > .05])
#     # plt.imshow(effect.reshape(6, 6), cmap='RdBu_r', extent=[1,
#     #            6, 1, 6], aspect='auto',
#     #            origin='lower')
#     plt.colorbar()
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Frequency (Hz)')
#     plt.title(r"Time-locked response for '%s' "% (effect_label) )
#     plt.show()


# plt.figure(figsize=(7,3))
# plt.title('Time-series of SD/LD in ATLs')
# for i in np.arange(0, len(X4)):
#     plt.plot(times, X4[i].mean(axis=0),my_color[i],label=labels[i])
 
# plt.xlabel("time (ms)")
# plt.ylabel("Amplitudes")
# plt.legend(loc='uuper left')
# plt.show()
# plt.savefig(C.pictures_path_Source_estimate+ 'time-series_SD-LD_ATLs.png')
  

###############################################################################

## Comparison of [SD_LD, Early_Late, Alpha_Beta_Gamma] with bar chart
# X_SD=C.ImCoh_SD.copy().reshape(2*3*18,6,6)
# X_LD=C.ImCoh_LD.copy().reshape(2*3*18,6,6)

# X_SDLD= [X_SD.copy().mean(0), X_LD.copy().mean(0)]
# means=[]
# errors=[]
# for i in np.arange(0,6):
#     for j in np.arange(i+1,6):
#         means.append( X_SDLD[0][j,i])
#         means.append( X_SDLD[1][j,i])

#         errors.append(np.std(X_SD[:,j,i]))
#         errors.append(np.std(X_LD[:,j,i]))


# mean = np.multiply(means,10**12)
# error = np.multiply(errors,10**12)

# my_color=['b','b','r','r','y','y','g','g','m','m','c','c','purple','purple','k','k',\
#           'b','b','r','r','y','y','g','g','m','m','c','c','purple','purple']

# labels = ['S1-2','L1-2', 'S1-3','L1-3','S1-4','L1-4','S1-5','L1-5','S1-6','L1-6','S2-3','L2-3','S2-4','L2-4','S2-5','L2-5','S2-6','L2-6',\
#           'S3-4', 'L3-4','S3-5','L3-5','S3-6','L3-6','S4-5','L4-5','S4-6','L4-6','S5-6','L5-6']
# x_pos = np.arange(len(labels))
# fig, ax = plt.subplots()
# for i in np.arange(0,30):
#     ax.bar(x_pos[i], means[i],
#             yerr=errors[i],
#             align='center',
#             alpha=0.5,
#             ecolor='black',
#             capsize=10,
#             color=my_color[i])

    
# ax.set_ylabel('Amplitude Mean (X e-12)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels)
# ax.set_title('Comparison of SD/LD in the left and right ATLs')
# # ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()
# plt.savefig(C.pictures_path_Source_estimate+ 'bar-chart_SD-LD_ATLs.png')




###############################################################
# plt.imshow(effect.reshape(6, 6), cmap='RdBu_r', extent=[1,
#     #            6, 1, 6], aspect='auto',
#     #            origin='lower')


# for win in np.arange(0, len(C.con_time_window)):
#     tmin = C.con_time_window[win]
#     tmax = C.con_time_window[win]+ C.con_time_window_len
#     for freq in np.arange(0, len(C.con_freq_band)-1):
  
#         fmin = C.con_freq_band[freq]
#         fmax = C.con_freq_band[freq+1]             
#         con_SD_LD['imcoh'] = C.ImCoh_SD_LD[win,freq,:,:]         
#         Con_SD['imcoh'] = C.ImCoh_SD_sorted[win,freq,:,:] 
#         Con_LD['imcoh'] = C.ImCoh_LD_sorted[win,freq,:,:] 
        
#         vmax_SD_LD = max(abs(C.ImCoh_SD_LD[win,freq,:,:].max()), abs(C.ImCoh_SD_LD[win,freq,:,:].min()))
#         vmax_SD = max(abs(C.ImCoh_SD[win,freq,:,:].max()), abs(C.ImCoh_SD[win,freq,:,:].min()))
#         vmax_LD = max(abs(C.ImCoh_LD[win,freq,:,:].max()), abs(C.ImCoh_LD[win,freq,:,:].min()))
#         # labels = mne.read_labels_from_annot( 'fsaverage', parc='aparc',
#         #                                     subjects_dir=C.data_path)
#         # unknwon = labels.pop(68)
        
#         # label_colors = [label.color for label in SN_ROI] 
#         label_colors = [(0.06,0.53,0.69,1),(0.06,0.53,0.69,1),\
#                         (0.02,0.83,0.62,1),(0.02,0.23,0.29,1),\
#                         (0.93,0.27,0.43,1),(1,0.81,0.4,1)]   

#         # label_names = [label.name for label in SN_ROI]
#         label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']
    
#         lh_labels = ['lATL','MTG_ITG','IFG','AG','PVA']
#         rh_labels = ['rATL']


#         node_order = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']
#         node_angles = circular_layout(label_names, node_order, start_pos=90,
#                                       group_boundaries=[0, len(label_names) / 2])
        
     
#         fig_con, axes_con = plot_connectivity_circle(Con_SD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) SD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')
            
#         fig_con, axes_con = plot_connectivity_circle(Con_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) LD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')                         
                                
#         fig_con, axes_con = plot_connectivity_circle(con_SD_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) SD-LD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD-LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')    
        
# # plt.close('all')
###############################################################
# Circular visualization

# con_SD_LD = dict()
# Con_SD = dict()
# Con_LD = dict()


# con_SD_LD['imcoh'] = C.ImCoh_SD_LD
# Con_SD['imcoh'] = C.ImCoh_SD_sorted
# Con_LD['imcoh'] = C.ImCoh_LD_sorted
# vmax_SD_LD = max(abs(C.ImCoh_SD_LD.max()), abs(C.ImCoh_SD_LD.min()))
# vmax_SD = max(abs(C.ImCoh_SD.max()), abs(C.ImCoh_SD.min()))
# vmax_LD = max(abs(C.ImCoh_LD.max()), abs(C.ImCoh_LD.min()))
# vmax = max(vmax_SD_LD,vmax_SD,vmax_LD)
# vmin = -vmax

# for win in np.arange(0, len(C.con_time_window)):
#     tmin = C.con_time_window[win]
#     tmax = C.con_time_window[win]+ C.con_time_window_len
#     for freq in np.arange(0, len(C.con_freq_band)-1):
  
#         fmin = C.con_freq_band[freq]
#         fmax = C.con_freq_band[freq+1]             
#         con_SD_LD['imcoh'] = C.ImCoh_SD_LD[win,freq,:,:]         
#         Con_SD['imcoh'] = C.ImCoh_SD_sorted[win,freq,:,:] 
#         Con_LD['imcoh'] = C.ImCoh_LD_sorted[win,freq,:,:] 
        
#         vmax_SD_LD = max(abs(C.ImCoh_SD_LD[win,freq,:,:].max()), abs(C.ImCoh_SD_LD[win,freq,:,:].min()))
#         vmax_SD = max(abs(C.ImCoh_SD[win,freq,:,:].max()), abs(C.ImCoh_SD[win,freq,:,:].min()))
#         vmax_LD = max(abs(C.ImCoh_LD[win,freq,:,:].max()), abs(C.ImCoh_LD[win,freq,:,:].min()))
#         # labels = mne.read_labels_from_annot( 'fsaverage', parc='aparc',
#         #                                     subjects_dir=C.data_path)
#         # unknwon = labels.pop(68)
        
#         # label_colors = [label.color for label in SN_ROI] 
#         label_colors = [(0.06,0.53,0.69,1),(0.06,0.53,0.69,1),\
#                         (0.02,0.83,0.62,1),(0.02,0.23,0.29,1),\
#                         (0.93,0.27,0.43,1),(1,0.81,0.4,1)]   

#         # label_names = [label.name for label in SN_ROI]
#         label_names = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']
    
#         lh_labels = ['lATL','MTG_ITG','IFG','AG','PVA']
#         rh_labels = ['rATL']


#         node_order = ['lATL','rATL','MTG_ITG','IFG','AG','PVA']
#         node_angles = circular_layout(label_names, node_order, start_pos=90,
#                                       group_boundaries=[0, len(label_names) / 2])
        
     
#         fig_con, axes_con = plot_connectivity_circle(Con_SD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) SD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')
            
#         fig_con, axes_con = plot_connectivity_circle(Con_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) LD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')                         
                                
#         fig_con, axes_con = plot_connectivity_circle(con_SD_LD['imcoh'], label_names, n_lines=None,\
#                           node_angles=node_angles, node_colors=label_colors,\
#                           title='Connectivity (ImCoh) SD-LD \n'+\
#                           f'{tmin}' +'-'+f'{tmax}'+'ms, '+f'{fmin}' +'-'+\
#                           f'{fmax}'+'Hz',facecolor='slategray',textcolor='white')  
#         fig_con.savefig(C.pictures_path_Source_estimate+ 'Connectivity(ImCoh)SD-LD_'+\
#                           f'{tmin:.3f}' +'-'+f'{tmax:.3f}'+'_'+f'{fmin}' +'-'+\
#                           f'{fmax}.jpg')    
        
# plt.close('all')
        