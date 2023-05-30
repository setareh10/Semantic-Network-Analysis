#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:21:53 2020

@author: sr05
"""


import numpy as np
import mne
import sn_config as C
import SN_semantic_ROIs 
from mne.minimum_norm import apply_inverse, read_inverse_operator
from scipy import stats
from matplotlib import pyplot as plt


 
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2

# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects




    
        
        
def stc_baseline_correction(X,tmin,tmax):
    time_dim = len(X.times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    # baseline_timepoints = X.times[np.where(X.times==tmin):np.where(X.times==tmax)]
    baseline_timepoints = X.times[0:300]

    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=1)
    corrected_stc = X - baseline_mean_mat
    return corrected_stc


import numpy as np
import mne
import sn_config as C
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
MRI_sub = C.subjects_mri_files

def SN_semantic_ROIs():
    # Loading Human Connectom Project parcellation
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=C.data_path,verbose=True)
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                        subjects_dir=C.data_path)
    
 
    ##............................. Control Regions ............................##
        
    # Temporal area - Splitting STSvp 
    label_STSvp = ['L_STSvp_ROI-lh']
    my_STSvp=[]
    for j in np.arange(0,len(label_STSvp )):
        my_STSvp.append([label for label in labels if label.name == \
                         label_STSvp[j]][0])
    
    for m in np.arange(0,len(my_STSvp)):
        if m==0:
            STSvp = my_STSvp[m]
        else:
            STSvp = STSvp + my_STSvp[m]
            
            
    [STSvp1,STSvp2,STSvp3,STSvp4,STSvp5,STSvp6]=mne.split_label(label=STSvp,parts\
        =('L_STSvp1_ROI-lh','L_STSvp2_ROI-lh','L_STSvp3_ROI-lh','L_STSvp4_ROI-lh',
          'L_STSvp5_ROI-lh','L_STSvp6_ROI-lh',),subject='fsaverage',subjects_dir=\
          C.data_path)
        
    # Temporal area - Splitting PH 
    label_PH = ['L_PH_ROI-lh']
    my_PH=[]
    for j in np.arange(0,len(label_PH )):
        my_PH.append([label for label in labels if label.name == label_PH[j]][0])
    
    for m in np.arange(0,len(my_PH)):
        if m==0:
            PH = my_PH[m]
        else:
            PH = PH + my_PH[m]
    
    [PH1,PH2]=mne.split_label(label=PH,parts=('L_PH1_ROI-lh','L_PH2_ROI-lh')\
              ,subject='fsaverage',subjects_dir=C.data_path)
    [PH21,PH22,PH23,PH24]=mne.split_label(label=PH2,parts=\
              ('L_PH21_ROI-lh','L_PH22_ROI-lh','L_PH23_ROI-lh','L_PH24_ROI-lh'),\
              subject='fsaverage',subjects_dir=C.data_path)
    
    
    # Temporal area - Splitting TE2p  
    label_TE2p = ['L_TE2p_ROI-lh']
    my_TE2p=[]
    for j in np.arange(0,len(label_TE2p )):
        my_TE2p.append([label for label in labels if label.name == label_TE2p[j]][0])
    
    for m in np.arange(0,len(my_TE2p)):
        if m==0:
            TE2p = my_TE2p[m]
        else:
            TE2p = TE2p + my_TE2p[m]
            
    
    [TE2p1,TE2p2]=mne.split_label(label=TE2p,parts=('L_TE2p1_ROI-lh',\
        'L_TE2p2_ROI-lh'),subject='fsaverage',subjects_dir=C.data_path)
    
    
    # Temporal area
    label_TE1p = ['L_TE1p_ROI-lh']
    my_TE1p=[]
    for j in np.arange(0,len(label_TE1p )):
        my_TE1p.append([label for label in labels if label.name == label_TE1p[j]][0])
    
    for m in np.arange(0,len(my_TE1p)):
        if m==0:
            TE1p = my_TE1p[m]
        else:
            TE1p = TE1p + my_TE1p[m]
            
    TG= STSvp1 + STSvp2 + STSvp3 + STSvp4 + TE2p1 + PH24 +TE1p
    
    
    
    ##.......................... Representation Regions .........................##
            
    # Left ATL area - splitting TE2a      
    label_TE2a = ['L_TE2a_ROI-lh']
    my_TE2a=[]
    for j in np.arange(0,len(label_TE2a )):
        my_TE2a.append([label for label in labels if label.name == label_TE2a[j]][0])
    
    for m in np.arange(0,len(my_TE2a)):
        if m==0:
            l_TE2a = my_TE2a[m]
        else:
            l_TE2a = l_TE2a + my_TE2a[m]
            
    
    [l_TE2a1,l_TE2a2,l_TE2a3]=mne.split_label(label=l_TE2a,parts=\
        ('L_TE2a1_ROI-lh','L_TE2a2_ROI-lh','L_TE2a3_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
        
        
    # Left ATL area - splitting TE1m 
    label_TE1m = ['L_TE1m_ROI-lh']
    my_TE1m=[]
    for j in np.arange(0,len(label_TE1m )):
        my_TE1m.append([label for label in labels if label.name == label_TE1m[j]][0])
    
    for m in np.arange(0,len(my_TE1m)):
        if m==0:
            l_TE1m = my_TE1m[m]
        else:
            l_TE1m = l_TE1m + my_TE1m[m]
            
    
    [l_TE1m1,l_TE1m2,l_TE1m3]=mne.split_label(label=l_TE1m,parts=\
        ('L_TE1m1_ROI-lh','L_TE1m2_ROI-lh','L_TE1m3_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)       
    [l_TE1m11,l_TE1m12,l_TE1m13]=mne.split_label(label=l_TE1m1,parts=\
        ('L_TE1m11_ROI-lh','L_TE1m12_ROI-lh','L_TE1m13_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    [l_TE1m21,l_TE1m22,l_TE1m23]=mne.split_label(label=l_TE1m2,parts=\
        ('L_TE1m21_ROI-lh','L_TE1m22_ROI-lh','L_TE1m23_ROI-lh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Left ATL area  
    label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE1a_ROI-lh']
    
    
    my_ATL=[]
    for j in np.arange(0,len(label_ATL )):
        my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])
    
    for m in np.arange(0,len(my_ATL)):
        if m==0:
            l_ATL = my_ATL[m]
        else:
            l_ATL = l_ATL + my_ATL[m]
            
    l_ATL = l_ATL + l_TE2a2 + l_TE2a3 + l_TE1m13 + l_TE1m23
    
    
    # Right ATL area - splitting TE2a      
    label_TE2a = ['R_TE2a_ROI-rh']
    my_TE2a=[]
    for j in np.arange(0,len(label_TE2a )):
        my_TE2a.append([label for label in labels if label.name == label_TE2a[j]][0])
    
    for m in np.arange(0,len(my_TE2a)):
        if m==0:
            r_TE2a = my_TE2a[m]
        else:
            r_TE2a = r_TE2a + my_TE2a[m]
            
    
    [r_TE2a1,r_TE2a2,r_TE2a3]=mne.split_label(label=r_TE2a,parts=\
        ('R_TE2a1_ROI-rh','R_TE2a2_ROI-rh','R_TE2a3_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Right ATL area - splitting TE1m 
    label_TE1m = ['R_TE1m_ROI-rh']
    my_TE1m=[]
    for j in np.arange(0,len(label_TE1m )):
        my_TE1m.append([label for label in labels if label.name == label_TE1m[j]][0])
    
    for m in np.arange(0,len(my_TE1m)):
        if m==0:
            r_TE1m = my_TE1m[m]
        else:
            r_TE1m = r_TE1m + my_TE1m[m]
            
    
    [r_TE1m1,r_TE1m2,r_TE1m3]=mne.split_label(label=r_TE1m,parts=\
        ('R_TE1m1_ROI-rh','R_TE1m2_ROI-rh','R_TE1m3_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)       
    
    [r_TE1m31,r_TE1m32,r_TE1m33]=mne.split_label(label=r_TE1m3,parts=\
        ('R_TE1m31_ROI-rh','R_TE1m32_ROI-rh','R_TE1m33_ROI-rh'),subject='fsaverage',\
        subjects_dir=C.data_path)
    
    # Right ATL area  
    label_ATL = ['R_TGd_ROI-rh','R_TGv_ROI-rh','R_TE1a_ROI-rh']
    
    
    my_ATL=[]
    for j in np.arange(0,len(label_ATL )):
        my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])
    
    for m in np.arange(0,len(my_ATL)):
        if m==0:
            r_ATL = my_ATL[m]
        else:
            r_ATL = r_ATL + my_ATL[m]
            
    r_ATL = r_ATL + r_TE2a2 + r_TE2a3 + r_TE1m33
    
    ## ............................ Angular Gyrus .............................. ##
    
    label_AG = ['L_PGi_ROI-lh','L_PGp_ROI-lh','L_PGs_ROI-lh']
    
    my_AG=[]
    for j in np.arange(0,len(label_AG)):
        my_AG.append([label for label in labels if label.name == label_AG[j]][0])
    
    for m in np.arange(0,len(my_AG )):
        if m==0:
            AG = my_AG[m]
        else:
            AG = AG  + my_AG[m]      
    
    
    ## ....................... Inferior Frontal Gyrus  ......................... ##
         
            
    label_IFG = ['L_44_ROI-lh','L_45_ROI-lh','L_47l_ROI-lh','L_p47r_ROI-lh']   
    my_IFG=[]
    for j in np.arange(0,len(label_IFG )):
        my_IFG.append([label for label in labels if label.name == label_IFG[j]][0])
    
    for m in np.arange(0,len(my_IFG)):
        if m==0:
            IFG = my_IFG[m]
        else:
            IFG = IFG + my_IFG[m] 
            
    ## ............................, Visual Area ............................... ##
    label_V1 = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh']        
    my_V1 =[]        
    for j in np.arange(0,len(label_V1 )):
        my_V1.append([label for label in labels if label.name == label_V1[j]][0])
    V1= my_V1[0]
    
                   
    SN_ROI =[l_ATL,r_ATL,TG,IFG,AG,V1]
    return SN_ROI


labels = SN_semantic_ROIs()



X_SD = np.zeros([18,6,1201])
X_LD = np.zeros([18,6,1201])

for i in np.arange(0, len(subjects)):
    n_subjects = len(subjects)
    meg = subjects[i]
    sub_to = MRI_sub[i]
    print('Participant : ' , i)
    
    # Reading epochs
    epo_name_SD = data_path + meg + 'block_SD_words_epochs-epo.fif'
    epo_name_LD = data_path + meg + 'block_LD_words_epochs-epo.fif'
        
    epochs_sd = mne.read_epochs(epo_name_SD, preload=True)
    epochs_ld = mne.read_epochs(epo_name_LD, preload=True)

    epochs_SD = epochs_sd['words'] 
    epochs_LD = epochs_ld['words'] 


    # Reading inverse operator
    inv_fname_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'

    inv_op_SD = read_inverse_operator(inv_fname_SD) 
    inv_op_LD = read_inverse_operator(inv_fname_LD) 
    
    # Evoked responses 
    evoked_SD = epochs_SD.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)
    evoked_LD = epochs_LD.average().set_eeg_reference(ref_channels = \
                        'average',projection=True)
                
    stc_sd = apply_inverse( evoked_SD, inv_op_SD,lambda2,method ='MNE', 
                          pick_ori="normal")
    stc_ld = apply_inverse( evoked_LD, inv_op_LD,lambda2,method ='MNE',
                            pick_ori="normal")
    stc_SD = stc_baseline_correction(stc_sd,-300, 0 ) 
    stc_LD = stc_baseline_correction(stc_ld,-300, 0 ) 


    src_SD = inv_op_SD['src']
    src_LD = inv_op_LD['src']
    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    morphed_labels = mne.morph_labels(labels,subject_to=data_path+sub_to,\
                     subject_from='fsaverage',subjects_dir=data_path)
        
    label_ts_SD = mne.extract_label_time_course(stc_SD, morphed_labels, src_SD,\
                  mode='mean_flip')       
    label_ts_LD = mne.extract_label_time_course(stc_LD, morphed_labels, src_LD,\
                  mode='mean_flip') 
        
    X_SD[i,:,:] = label_ts_SD
    X_LD[i,:,:] = label_ts_LD

    # for i in np.arange(0,len(labels)-5):    
my_colors = ['b', 'r','y','g','black','c']
labless=['lATL','rATL','TG','IFG','AG','V']
t_value , p_value = stats.ttest_rel(X_SD,X_LD)
y=np.arange(0,600)
fig, ax = plt.subplots(4,figsize=(10, 40))
# for n in np.arange(0,len(labels)):
for n in np.array([0,1,2,3]):

    ax[0].plot(y,  X_SD.copy().mean(0)[n,300:900],my_colors[n],label=labless[n])
    ax[1].plot(y , X_LD.copy().mean(0)[n,300:900],my_colors[n])
    ax[2].plot(y , X_SD.copy().mean(0)[n,300:900]-X_LD.copy().mean(0)[n,300:900],\
               my_colors[n])
    ax[3].plot(y , t_value[n,300:900],my_colors[n])
 
    ax[0].legend()

    ax[0].set_ylabel('SD')
    ax[1].set_ylabel('LD')
    ax[2].set_ylabel('SD - LD')
    ax[3].set_ylabel('t-value')