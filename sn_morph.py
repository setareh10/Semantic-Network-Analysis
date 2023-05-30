#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:45:41 2020

@author: sr05
"""


import mne
import numpy as np
import sn_config as C
from mne.minimum_norm import read_inverse_operator 


# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
# subjects' directories
subjects =  C.subjects_mri
 
for i in np.arange(0, len(subjects)):
    print('participant : ' , i , ' / EEG + MEG')
    subject_from = subjects[i]
    meg = subjects[i][5:]
    # fname_morph = C.fname_STC(C, 'SensitivityMaps', subject, 'SensMap_' + modality + '_' + metric + '_mph')

     ##................................EEG + MEG............................##

    inv_fname_EMEG_SD = data_path + meg + 'InvOp_SD_EMEG-inv.fif'
    inv_fname_EMEG_LD = data_path + meg + 'InvOp_LD_EMEG-inv.fif'


    inv_op_SD = read_inverse_operator(inv_fname_EMEG_SD)   
    inv_op_LD = read_inverse_operator(inv_fname_EMEG_LD)    
    
    stc_fname_SD_words  = data_path + meg + 'block_SD_words_EMEG'
    stc_fname_LD_words  = data_path + meg + 'block_LD_words_EMEG'

    stc_SD_words = mne.read_source_estimate(stc_fname_SD_words)
    stc_LD_words = mne.read_source_estimate(stc_fname_LD_words)

    # # setup source morph
    morph_SD_words = mne.compute_source_morph( src=inv_op_SD['src'], 
                  subject_from = stc_SD_words.subject, 
                  subject_to = C.subject_to , spacing = C.spacing_morph, 
                  subjects_dir = C.data_path)
    morph_LD_words = mne.compute_source_morph( src=inv_op_LD['src'], 
                  subject_from = stc_LD_words.subject, 
                  subject_to = C.subject_to , spacing = C.spacing_morph, 
                  subjects_dir = C.data_path)

    stc_SD_words_fsaverage = morph_SD_words.apply(stc_SD_words)
    stc_LD_words_fsaverage = morph_LD_words.apply(stc_LD_words)

    fname_SD_words_fsaverage = C.data_path + meg + 'block_SD_words_EMEG_fsaverage'
    fname_LD_words_fsaverage = C.data_path + meg + 'block_LD_words_EMEG_fsaverage'


    stc_SD_words_fsaverage.save(fname_SD_words_fsaverage)
    stc_LD_words_fsaverage.save(fname_LD_words_fsaverage)
