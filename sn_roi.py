#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:55:20 2020

@author: sr05
"""


import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from mne.epochs import equalize_epoch_counts
import sn_config as C
from surfer import Brain
# path to raw data
data_path = C.data_path
main_path = C.main_path
subjects =  C.subjects
# Parameters
snr = C.snr
lambda2 = C.lambda2


mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=C.data_path,verbose=True)
        

labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',\
                                    subjects_dir=C.data_path)
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
brain.add_annotation('HCPMMP1')
    
# label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TF_ROI-lh','L_TE2a_ROI-lh','L_TE2p_ROI-lh' ,'L_TE1a_ROI-lh','L_TE1m_ROI-lh']

label_ATL = ['L_TGd_ROI-lh','L_TGv_ROI-lh','L_TE2a_ROI-lh','L_TE1a_ROI-lh','L_TE1m_ROI-lh']

#label_ATL = ['L_TE2a_ROI-lh','L_TE1a_ROI-lh','L_TE1m_ROI-lh']

my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]
        
 
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))

for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False)
        

#label_ATL = ['L_MT_ROI-lh','L_MST_ROI-lh','L_V4t_ROI-lh','L_FST_ROI-lh',
#             'L_LO1_ROI-lh','L_LO2_ROI-lh','L_LO3_ROI-lh','L_PH_ROI-lh']
#label_ATL = ['L_A4_ROI-lh','L_A5_ROI-lh','L_STSda_ROI-lh','L_STSdp_ROI-lh',
#             'L_STSva_ROI-lh','L_STSvp_ROI-lh','L_STGa_ROI-lh','L_TA2_ROI-lh']
label_ATL = ['L_STSvp_ROI-lh', 'L_TPOJ1_ROI-lh','L_PH_ROI-lh','L_TE1p_ROI-lh','L_TE2p_ROI-lh']
my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]
for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False)



label_ATL = [ 'L_TPOJ1_ROI-lh','L_PH_ROI-lh','L_TE1p_ROI-lh','L_TE2p_ROI-lh']
my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]
for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False)
        
 
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))

for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False) 
    
    
    

label_ITG = ['L_PH_ROI-lh']


my_ITG=[]
for j in np.arange(0,len(label_ITG )):
    my_ITG.append([label for label in labels if label.name == label_ITG[j]][0])

for m in np.arange(0,len(my_ITG)):
    if m==0:
        ITG = my_ITG[m]
    else:
        ITG = ITG + my_ITG[m]
        
 
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))

for m in np.arange(0,len(my_ITG)):
    
    brain.add_label(my_ITG[m], borders=False)
        
    
        
label_IFG = ['L_44_ROI-lh','L_45_ROI-lh','L_47l_ROI-lh']   
my_IFG=[]
for j in np.arange(0,len(label_IFG )):
    my_IFG.append([label for label in labels if label.name == label_IFG[j]][0])

for m in np.arange(0,len(my_IFG)):
    if m==0:
        IFG = my_IFG[m]
    else:
        IFG = IFG + my_IFG[m] 
        
        
label_V1 = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh']        
my_V1 =[]        
for j in np.arange(0,len(label_V1 )):
    my_V1.append([label for label in labels if label.name == label_V1[j]][0])
V1= my_V1[0]


for m in np.arange(0,len(my_V1)):
    
    brain.add_label(my_V1[m], borders=False)
    
    
    



label_MTG = ['L_TE1p_ROI-lh','L_TE2p_ROI-lh']   
my_MTG=[]
for j in np.arange(0,len(label_MTG)):
    my_MTG.append([label for label in labels if label.name == label_MTG[j]][0])

for m in np.arange(0,len(my_MTG)):
    if m==0:
        MTG = my_MTG[m]
    else:
        MTG = MTG + my_MTG[m] 
        
for m in np.arange(0,len(my_MTG)):
    
    brain.add_label(my_MTG[m], borders=False)
            
        
        
        
        
label_SPC = ['L_7Pm_ROI-lh','L_7PL_ROI-lh','L_7Am_ROI-lh','L_7AL_ROI-lh',
               'L_7PC_ROI-lh' ,'L_LIPv_ROI-lh','L_LIPd_ROI-lh','L_MIP_ROI-lh','L_AIP_ROI-lh',
               'L_VIP_ROI-lh']

my_SPC=[]
for j in np.arange(0,len(label_SPC)):
    my_SPC.append([label for label in labels if label.name == label_SPC[j]][0])

for m in np.arange(0,len(my_SPC )):
    if m==0:
        SPC = my_SPC[m]
    else:
        SPC = SPC  + my_SPC[m]       
        
        
#label_IPC = ['L_PGp_ROI-lh','L_IP0_ROI-lh','L_IP1_ROI-lh','L_IP2_ROI-lh',
#               'L_PF_ROI-lh' ,'L_PFt_ROI-lh','L_PFop_ROI-lh','L_PFm_ROI-lh',
#               'L_PGi_ROI-lh','L_PGs_ROI-lh']

# label_IPC = ['L_PFm_ROI-lh','L_PGi_ROI-lh','L_PGs_ROI-lh']
label_IPC = ['L_PGi_ROI-lh']


my_IPC=[]
for j in np.arange(0,len(label_IPC)):
    my_IPC.append([label for label in labels if label.name == label_IPC[j]][0])

for m in np.arange(0,len(my_IPC )):
    if m==0:
        IPC = my_IPC[m]
    else:
        IPC = IPC  + my_IPC[m]           
        
for m in np.arange(0,len(my_IPC)):
    
    brain.add_label(my_IPC[m], borders=False)


        
#label_VV = ['L_V8_ROI-lh','L_VVC_ROI-lh','L_PIT_ROI-lh','L_FFC_ROI-lh',
#               'L_VMV1_ROI-lh' ,'L_VMV2_ROI-lh','L_VMV3_ROI-lh']
##label_VV = ['L_FFC_ROI-lh']
#
#my_VV=[]
#for j in np.arange(0,len(label_VV)):
#    my_VV.append([label for label in labels if label.name == label_VV[j]][0])
#
#for m in np.arange(0,len(my_VV )):
#    if m==0:
#        VV = my_VV[m]
#    else:
#        VV= VV  + my_VV[m]            
        
        
          
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400)) 

label_ATL = ['L_STSvp_ROI-lh', 'L_PH_ROI-lh','L_TE1p_ROI-lh','L_TE2p_ROI-lh']
my_ATL=[]
for j in np.arange(0,len(label_ATL )):
    my_ATL.append([label for label in labels if label.name == label_ATL[j]][0])

for m in np.arange(0,len(my_ATL)):
    if m==0:
        ATL = my_ATL[m]
    else:
        ATL = ATL + my_ATL[m]
for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=False)

       

brain.add_label(ATL, borders=False,color='green')
brain.add_label(V1, borders=False, color='yellow')
brain.add_label(MTG, borders=False, color='green')
#brain.add_label(SPC, borders=False, color='orange')
brain.add_label(IPC, borders=False, color='purple')
brain.add_label(IFG, borders=False,color='red')
for m in np.arange(0,len(my_IPC)):
    
    brain.add_label(my_IPC[m], borders=True)




#                                ,'L_TE1p_ROI-lh',
#                           'L_PHT_ROI-lh','L_TPOJ1_ROI-lh','L_TPOJ2_ROI-lh','L_TPOJ3_ROI-lh',
#                           'L_PSL_ROI-lh','L_STV_ROI-lh','L_44_ROI-lh','L_45_ROI-lh',
#                           'L_47l_ROI-lh','L_IFJa_ROI-lh','L_IFJp_ROI-lh','L_IFSa_ROI-lh',
#                           'L_IFSp_ROI-lh',
#                           'R_TGd_ROI-rh','R_TGv_ROI-rh','R_TF_ROI-rh','R_TE2a_ROI-rh',
#                           'R_TE2p_ROI-rh','R_TE1a_ROI-rh','R_TE1m_ROI-rh','R_TE1p_ROI-rh',
#                           'R_PHT_ROI-rh','R_TPOJ1_ROI-rh','R_TPOJ2_ROI-rh','R_TPOJ3_ROI-rh',
#                           'R_PSL_ROI-rh','R_STV_ROI-rh','R_44_ROI-rh','R_45_ROI-rh',
#                           'R_47l_ROI-rh','R_IFJa_ROI-rh','R_IFJp_ROI-rh','R_IFSa_ROI-rh',
#                           'R_IFSp_ROI-rh']
##            
           
            
          
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
brain.add_annotation('HCPMMP1')


aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]




        
brain.add_label(ATL, borders=False)

for m in np.arange(0,len(my_ATL)):
    
    brain.add_label(my_ATL[m], borders=True)

#*********************
label_STSvp = ['L_STSvp_ROI-lh']
my_STSvp=[]
for j in np.arange(0,len(label_STSvp )):
    my_STSvp.append([label for label in labels if label.name == label_STSvp[j]][0])


label_TPOJ1 = ['L_TPOJ1_ROI-lh']
my_TPOJ1=[]
for j in np.arange(0,len(label_TPOJ1 )):
    my_TPOJ1.append([label for label in labels if label.name == label_TPOJ1[j]][0])
    
    
label_PH = ['L_PH_ROI-lh']
my_PH=[]
for j in np.arange(0,len(label_PH )):
    my_PH.append([label for label in labels if label.name == label_PH[j]][0])
    
    
label_TE1p = ['L_TE1p_ROI-lh']
my_TE1p=[]
for j in np.arange(0,len(label_TE1p )):
    my_TE1p.append([label for label in labels if label.name == label_TE1p[j]][0])
    
label_TE2p = ['L_TE2p_ROI-lh']
my_TE2p=[]
for j in np.arange(0,len(label_TE2p )):
    my_TE2p.append([label for label in labels if label.name == label_TE2p[j]][0])
for m in np.arange(0,len(my_TE2p)):
    
    brain.add_label(my_TE2p[m], borders=False)
    
    
    
brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))        

brain.add_label(my_STSvp, borders=False,color='blue')
brain.add_label(my_TPOJ1, borders=True, color='yellow')
brain.add_label(my_PH, borders=False, color='green')
brain.add_label(my_TE1p, borders=False, color='red')
brain.add_label(my_TE2p, borders=True, color='purple')




 

#label_O = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh','L_V3A_ROI-lh',\
#           'L_V3B_ROI-lh','L_V6_ROI-lh','L_V6A_ROI-lh','L_V7_ROI-lh',]


label_O = ['L_V1_ROI-lh','L_V2_ROI-lh','L_V3_ROI-lh','L_V4_ROI-lh','L_V3A_ROI-lh',\
           'L_V3B_ROI-lh','L_V6A_ROI-lh','L_V7_ROI-lh','L_V3CD_ROI-lh',\
           'L_LO1_ROI-lh','L_LO2_ROI-lh','L_LO3_ROI-lh']#,\

#label_O = ['L_V3CD_ROI-lh']#,\

#           'L_MT_ROI-lh' ,'L_MST_ROI-lh','L_V4t_ROI-lh','L_FST_ROI-lh']

brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=C.data_path,
              cortex='low_contrast', background='white', size=(400, 400))
my_O=[]

for j in np.arange(0,len(label_O )):
    my_O.append([label for label in labels if label.name == label_O[j]][0])

for m in np.arange(0,len(my_O)):
    if m==0:
        O = my_O[m]
    else:
        O = O + my_O[m]
        
for m in np.arange(0,len(my_O)):
    
    brain.add_label(my_O[m], borders=False, color='purple')
    
    
    
for m in np.arange(0,len(my_AG)):
    
    brain.add_label(my_AG[m], borders=False, color='orange')