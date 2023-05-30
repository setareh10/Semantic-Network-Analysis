#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 01:11:55 2020

@author: sr05

place it in same folder as py-files
"""

##########################################################
## SYSTEM variables
##########################################################

import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as stats


##########################################################
## GENERAL
##########################################################

# path to maxfiltered raw data
main_path = '/imaging/hauk/users/rf02/Semnet/'

# path to filtered raw data
data_path = '/imaging/hauk/users/sr05/SemNet/SemNetData/'

label_path = '/imaging/hauk/users/rf02/TypLexMEG/fsaverage/label'

# subjects' EMEG directories
subjects = [
    # '/meg16_0030/160216/',#0
    '/meg16_0032/160218/',  # 1
    '/meg16_0034/160219/',  # 2
    '/meg16_0035/160222/',  # 3
    '/meg16_0042/160229/',  # 4
    '/meg16_0045/160303/',  # 5
    '/meg16_0052/160310/',  # 6
    '/meg16_0056/160314/',  # 7
    '/meg16_0069/160405/',  # 8
    '/meg16_0070/160407/',  # 9
    '/meg16_0072/160408/',  # 10
    '/meg16_0073/160411/',  # 11
    '/meg16_0075/160411/',  # 12
    '/meg16_0078/160414/',  # 13
    '/meg16_0082/160418/',  # 14
    '/meg16_0086/160422/',  # 15
    '/meg16_0097/160512/',  # 16
    '/meg16_0122/160707/',  # 17
    '/meg16_0125/160712/',  # 18
]
# subjects' MRI directories
subjects_mri = [
    # '/MRI_meg16_0030/160216/',#0
    '/MRI_meg16_0032/160218/',  # 1
    '/MRI_meg16_0034/160219/',  # 2
    '/MRI_meg16_0035/160222/',  # 3
    '/MRI_meg16_0042/160229/',  # 4
    '/MRI_meg16_0045/160303/',  # 5
    '/MRI_meg16_0052/160310/',  # 6
    '/MRI_meg16_0056/160314/',  # 7
    '/MRI_meg16_0069/160405/',  # 8
    '/MRI_meg16_0070/160407/',  # 9
    '/MRI_meg16_0072/160408/',  # 10
    '/MRI_meg16_0073/160411/',  # 11
    '/MRI_meg16_0075/160411/',  # 12
    '/MRI_meg16_0078/160414/',  # 13
    '/MRI_meg16_0082/160418/',  # 14
    '/MRI_meg16_0086/160422/',  # 15
    '/MRI_meg16_0097/160512/',  # 16
    '/MRI_meg16_0122/160707/',  # 17
    '/MRI_meg16_0125/160712/',  # 18
]

subjects_mri_files = [
    # '/MRI_0030/',#0
    '/MRI_0032/',  # 1
    '/MRI_0034/',  # 2
    '/MRI_0035/',  # 3
    '/MRI_0042/',  # 4
    '/MRI_0045/',  # 5
    '/MRI_0052/',  # 6
    '/MRI_0056/',  # 7
    '/MRI_0069/',  # 8
    '/MRI_0070/',  # 9
    '/MRI_0072/',  # 10
    '/MRI_0073/',  # 11
    '/MRI_0075/',  # 12
    '/MRI_0078/',  # 13
    '/MRI_0082/',  # 14
    '/MRI_0086/',  # 15
    '/MRI_0097/',  # 16
    '/MRI_0122/',  # 17
    '/MRI_0125/',  # 18
]

##########################################################
## MAXFILTER
##########################################################
cbu_path = '/megdata/cbu/semnet/'
# Path to the FIF file with cross-talk correction information.
cross_talk = '/neuro/databases_vectorview/ctc/ct_sparse.fif'

# Path to the '.dat' file with fine calibration coefficients
calibration = '/neuro/databases_vectorview/sss/sss_cal.dat'

# Dictionaries to write MEG bad channels
meg_bad_channels_fruit = {}
meg_bad_channels_odour = {}
meg_bad_channels_milk = {}
meg_bad_channels_ld = {}

destination_files = [
    # 'block2_milk.fif',     #0
    'block_fruit_raw.fif',  # 1
    'block_odour_raw.fif',  # 2
    'block_fruit_raw.fif',  # 3
    'block_milk_raw.fif',  # 4
    'Block_odour_aw.fif',  # 5
    'block_odour_raw.fif',  # 6
    'block_fruit_raw.fif',  # 7
    'block_odour_raw.fif',  # 8
    'block_odour_raw.fif',  # 9
    'block_odour_raw.fif',  # 10
    'block_milk_raw.fif',  # 11
    'block_fruit_raw.fif',  # 12
    'block_odour_raw.fif',  # 13
    'block_odour_raw.fif',  # 14
    'block_milk_raw.fif',  # 15
    'block_fruit_raw.fif',  # 16
    'block_milk_raw.fif',  # 17
    'block_fruit_raw.fif'  # 18
]
##########################################################
## Band Pass Filter
##########################################################

# filter parameters
l_freq = 0.1
h_freq = 45

# EEG bad channels

eeg_bad_channels_fruit = [
    ['EEG034'],
    ['EEG045', 'EEG002', 'EEG008', 'EEG004', 'EEG005', 'EEG009', 'EEG010'],
    ['EEG059', 'EEG074', 'EEG073', 'EEG072', 'EEG071', 'EEG066'],
    ['EEG071', 'EEG069'],
    ['EEG055', 'EEG072', 'EEG071', 'EEG002'],
    ['EEG039', 'EEG058'],
    [],
    ['EEG001'],
    ['EEG043', 'EEG057', 'EEG046'],
    ['EEG043', 'EEG058', 'EEG047', 'EEG054', 'EEG046'],
    ['EEG066', 'EEG039'],
    ['EEG071', 'EEG072'],
    ['EEG073', 'EEG071', 'EEG068', 'EEG072'],
    ['EEG039', 'EEG059', 'EEG074', 'EEG070', 'EEG072', 'EEG029'],
    ['EEG057'],
    ['EEG071', 'EEG068', 'EEG039'],
    ['EEG013'],
    ['EEG067'],
    ['EEG035']
]

eeg_bad_channels_odour = [
    ['EEG008', 'EEG028', 'EEG034'],
    ['EEG045', 'EEG002', 'EEG008', 'EEG009', 'EEG010', 'EEG027'],
    ['EEG071', 'EEG066', 'EEG069'],
    ['EEG057', 'EEG069', 'EEG071', 'EEG067'],
    ['EEG071', 'EEG002'],
    ['EEG034', 'EEG039', 'EEG058'],
    [],
    ['EEG001'],
    ['EEG043', 'EEG046', 'EEG057', 'EEG068', 'EEG071'],
    ['EEG072', 'EEG053', 'EEG058', 'EEG046', 'EEG047', 'EEG054'],
    ['EEG066', 'EEG039'],
    ['EEG072', 'EEG071'],
    ['EEG073', 'EEG068', 'EEG071', 'EEG072'],
    ['EEG039', 'EEG029', 'EEG074', 'EEG059', 'EEG073', 'EEG072'],
    ['EEG057', 'EEG047'],
    ['EEG071', 'EEG068', 'EEG072', 'EEG073', 'EEG074', 'EEG039'],
    ['EEG013'],
    ['EEG067'],
    ['EEG035']
]

eeg_bad_channels_milk = [
    ['EEG034'],
    ['EEG045', 'EEG071', 'EEG009', 'EEG010', 'EEG027'],
    ['EEG059', 'EEG073', 'EEG072', 'EEG066', 'EEG071'],
    ['EEG069', 'EEG071', 'EEG067'],
    ['EEG071', 'EEG072', 'EEG073', 'EEG002'],
    ['EEG039'],
    [],
    ['EEG071'],
    ['EEG045', 'EEG043', 'EEG056', 'EEG046', 'EEG048'],
    ['EEG058', 'EEG072', 'EEG071', 'EEG053', 'EEG047', 'EEG046', 'EEG054',
     'EEG034'],
    ['EEG039', 'EEG066', 'EEG071'],
    ['EEG071', 'EEG072'],
    ['EEG002', 'EEG073', 'EEG071', 'EEG072', 'EEG068'],
    ['EEG039', 'EEG074', 'EEG070', 'EEG069', 'EEG029'],
    ['EEG058', 'EEG057', 'EEG047', 'EEG046'],
    ['EEG034', 'EEG039', 'EEG071', 'EEG039'],
    [],
    ['EEG067'],
    ['EEG035']
]

eeg_bad_channels_ld = [
    ['EEG034', 'EEG053', 'EEG046', 'EEG067', 'EEG070', 'EEG071'],
    ['EEG048', 'EEG045', 'EEG027', 'EEG009', 'EEG010', 'EEG030'],
    ['EEG010', 'EEG072', 'EEG071', 'EEG069', 'EEG073', 'EEG074',
     'EEG060', 'EEG059', 'EEG066'],
    ['EEG071', 'EEG069', 'EEG068', 'EEG070', 'EEG073'],
    ['EEG072', 'EEG071', 'EEG074', 'EEG073', 'EEG070'],
    ['EEG034', 'EEG058', 'EEG071', 'EEG066', 'EEG002', 'EEG004', 'EEG024', ],
    ['EEG067'],
    ['EEG069', 'EEG074'],
    ['EEG043', 'EEG057', 'EEG047', 'EEG046'],
    ['EEG019', 'EEG053', 'EEG072', 'EEG071', 'EEG004', 'EEG066'],
    ['EEG066'],
    ['EEG072', 'EEG071'],
    ['EEG073', 'EEG071', 'EEG068', 'EEG072'],
    ['EEG027', 'EEG016', 'EEG074', 'EEG073', 'EEG029', 'EEG039'],
    ['EEG047'],
    ['EEG034', 'EEG071', 'EEG039'],
    ['EEG013'],
    ['EEG067'],
    ['EEG035', 'EEG032', 'EEG069', 'EEG071', 'EEG067']
]

# eeg_bad_channels = [
#     ['EEG008', 'EEG028'],  # 0
#     ['EEG067'],  # 1
#     ['EEG027', 'EEG028'],  # 2
#     ['EEG013', 'EEG038', 'EEG039', 'EEG073'],  # 3
#     ['EEG003', 'EEG004', 'EEG022', 'EEG023', 'EEG037', 'EEG038',
#      'EEG045', 'EEG046', 'EEG059', 'EEG072'],  # 4
#     ['EEG002', 'EEG034', 'EEG045', 'EEG046'],  # 5
#     ['EEG023', 'EEG034', 'EEG039', 'EEG041', 'EEG047'],  # 6
#     ['EEG003', 'EEG007', 'EEG008', 'EEG027', 'EEG046', 'EEG067',
#      'EEG070'],  # 7
#     ['EEG020', 'EEG055'],  # 8
#     ['EEG044', 'EEG045', 'EEG055', 'EEG057', 'EEG059', 'EEG060'],  # 9
#     ['EEG038', 'EEG039', 'EEG073'],  # 10
#     ['EEG044', 'EEG045'],  # 11
#     ['EEG002', 'EEG045', 'EEG046'],  # 12
#     ['EEG029', 'EEG039', 'EEG067'],  # 13
#     ['EEG033', 'EEG034', 'EEG044', 'EEG045', 'EEG046'],  # 14
#     ['EEG039', 'EEG045'],  # 15
#     [],  # 16
#     [],  # 17
#     ['EEG033']  # 18
# ]

# MEG bad channels based on Maxwell filter
meg_bad_channels_fruit = [
    ['MEG0141', 'MEG1142', 'MEG1731'],
    ['MEG1533', 'MEG1731'],
    ['MEG1731'],
    ['MEG1211', 'MEG1731', 'MEG1743', 'MEG2222', 'MEG2523', 'MEG1542'],
    ['MEG0813', 'MEG1731'],
    ['MEG1731'],
    ['MEG0413', 'MEG0813', 'MEG1111', 'MEG1731', 'MEG2323'],
    ['MEG0343', 'MEG1731', 'MEG2212'],
    ['MEG0933', 'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1621',
     'MEG1633', 'MEG1731'],
    ['MEG1731'],
    ['MEG1731', 'MEG2132', 'MEG2323'],
    ['MEG1731', 'MEG2122'],
    ['MEG1412', 'MEG1423', 'MEG1433', 'MEG1731', 'MEG2223'],
    ['MEG0813', 'MEG1731', 'MEG2033', 'MEG2512'],
    ['MEG1533', 'MEG1531', 'MEG1713', 'MEG1712', 'MEG1711', 'MEG1731',
     'MEG2323'],
    ['MEG0442', 'MEG1731', 'MEG2211'],
    ['MEG1731', 'MEG2323', 'MEG2513', 'MEG2511', 'MEG1412'],
    ['MEG1211', 'MEG1731', 'MEG2323', 'MEG2042'],
    ['MEG1211', 'MEG1731'],
]

meg_bad_channels_odour = [
    ['MEG0813', 'MEG1142', 'MEG1731'],
    ['MEG1543', 'MEG1542', 'MEG1731', 'MEG1541'],
    ['MEG1731'],
    ['MEG0631', 'MEG1611', 'MEG1731', 'MEG1743', 'MEG2323', 'MEG1542',
     'MEG2523'],
    ['MEG0813', 'MEG1731'],
    ['MEG0923', 'MEG1731'],
    ['MEG0413', 'MEG0631', 'MEG0813', 'MEG1111', 'MEG1731'],
    ['MEG0343', 'MEG0723', 'MEG1731', 'MEG2212', 'MEG2323'],
    ['MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1621', 'MEG1633',
     'MEG1631', 'MEG1643', 'MEG1731', 'MEG2033'],
    ['MEG0813', 'MEG1731'],
    ['MEG0923', 'MEG1731', 'MEG2323'],
    ['MEG1731', 'MEG2122', 'MEG2121'],
    ['MEG1412', 'MEG1423', 'MEG1433', 'MEG1442', 'MEG1731', 'MEG2141',
     'MEG2223', 'MEG2323'],
    ['MEG1731', 'MEG2033', 'MEG2512'],
    ['MEG1533', 'MEG1713', 'MEG1711', 'MEG1731', 'MEG2323', 'MEG2633',
     'MEG2631'],
    ['MEG0442', 'MEG1731'],
    ['MEG1611', 'MEG1731', 'MEG2323', 'MEG2511', 'MEG1412'],
    ['MEG1211', 'MEG1731', 'MEG2323', 'MEG2042'],
    ['MEG1122', 'MEG1211', 'MEG1731'],
]

meg_bad_channels_milk = [
    ['MEG1142', 'MEG1731'],
    ['MEG1533', 'MEG1731'],
    ['MEG1731'],
    ['MEG1211', 'MEG1731', 'MEG1743', 'MEG2523', 'MEG1542'],
    ['MEG1731'],
    ['MEG1731'],
    ['MEG0413', 'MEG1731'],
    ['MEG0343', 'MEG0723', 'MEG1731', 'MEG2212', 'MEG2511'],
    ['MEG1613', 'MEG1612', 'MEG1611', 'MEG1622', 'MEG1623', 'MEG1621',
     'MEG1632', 'MEG1633', 'MEG1631', 'MEG1643', 'MEG1642', 'MEG1641',
     'MEG1731'],
    ['MEG0813', 'MEG1731'],
    ['MEG1611', 'MEG1731', 'MEG2132', 'MEG2142', 'MEG2323'],
    ['MEG1731', 'MEG2113', 'MEG2122', 'MEG2121'],
    ['MEG0813', 'MEG1412', 'MEG1423', 'MEG1433', 'MEG1731', 'MEG1941',
     'MEG2223'],
    ['MEG1731', 'MEG2033'],
    ['MEG1533', 'MEG1531', 'MEG1713', 'MEG1711', 'MEG1731', 'MEG2533'],
    ['MEG0442', 'MEG1731', 'MEG2211'],
    ['MEG0813', 'MEG1731', 'MEG2323', 'MEG2513', 'MEG2511', 'MEG1412'],
    ['MEG1211', 'MEG1731', 'MEG2323', 'MEG2042'],
    ['MEG1211', 'MEG1731', 'MEG2323'],
]

meg_bad_channels_ld = [
    ['MEG1142', 'MEG1731', 'MEG2323'],
    ['MEG0813', 'MEG1211', 'MEG1533', 'MEG1731'],
    ['MEG1543', 'MEG1731', 'MEG1541', 'MEG1743'],
    ['MEG1211', 'MEG1731', 'MEG1743', 'MEG2523', 'MEG1542'],
    ['MEG0813', 'MEG1731'],
    ['MEG1731'],
    ['MEG0413', 'MEG0813', 'MEG1731'],
    ['MEG0723', 'MEG1731', 'MEG2212', 'MEG2311', 'MEG2323'],
    ['MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1621', 'MEG1632',
     'MEG1633', 'MEG1631', 'MEG1643', 'MEG1642', 'MEG1731', 'MEG2033'],
    ['MEG1731', 'MEG2331'],
    ['MEG0813', 'MEG1611', 'MEG1731', 'MEG1841', 'MEG2143', 'MEG2141'],
    ['MEG1731', 'MEG2113', 'MEG2122', 'MEG2121'],
    ['MEG0813', 'MEG1412', 'MEG1423', 'MEG1433', 'MEG1442', 'MEG1731',
     'MEG2141', 'MEG2223', 'MEG2623', 'MEG2633'],
    ['MEG0813', 'MEG1731', 'MEG2332', 'MEG2333', 'MEG2331', 'MEG2512',
     'MEG2513', 'MEG2511', 'MEG2523', 'MEG2533', 'MEG2542'],
    ['MEG1533', 'MEG1531', 'MEG1713', 'MEG1711', 'MEG1731', 'MEG2323',
     'MEG2533', 'MEG2532', 'MEG2543', 'MEG2541', 'MEG2633', 'MEG2631'],
    ['MEG0442', 'MEG0633', 'MEG0631', 'MEG1731', 'MEG2211'],
    ['MEG1533', 'MEG1731', 'MEG2323', 'MEG2322', 'MEG2513', 'MEG2511',
     'MEG2522', 'MEG1412'],
    ['MEG1731', 'MEG1922', 'MEG2323', 'MEG2641', 'MEG2042'],
    ['MEG1211', 'MEG1731'],
]
##########################################################
### EOG Removal
##########################################################
# ICA parameters
n_components = .95
ica_method = 'fastica'
decim = 3
n_max_eog = 2  # EOG061/EOG062

##########################################################
### ECG Removal
##########################################################
# ICA parameters
n_components = .95
method = 'fastica'
decim = 3
n_max_ecg = 1  # ECG063

##########################################################
### EPOCHING + Evoked 
##########################################################
# Events info
event_id_sd = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
               'pwordc': 6, 'target': 8}
event_id_ld = {'visual': 1, 'hear': 2, 'hand': 3, 'neutral': 4, 'emotional': 5,
               'pwordc': 6, 'pworda': 7, 'filler': 9}

# Parameters
tmin = -0.3
tmax = 0.9
stim_delay = 0.034  # delay in s
category_code = list(range(1, 6))

# baseline
epo_baseline = (tmin, 0.)

# artefact rejection thresholds
reject = dict(eeg=150e-6, grad=200e-12, mag=4e-12)

evoked_fruit_categories = {}
evoked_odour_categories = {}
evoked_milk_categories = {}
evoked_ld_categories = {}

##########################################################
### Grand Average
##########################################################
all_evokeds_sd_words = []
all_evokeds_ld_words = []

all_sd_words_nave = 0
all_ld_words_nave = 0
plot_peaks = ['eeg', 'mag', 'grad']

ts_args = dict(gfp=True)

topomap_args = dict(sensors=True)

##########################################################
### COVARIANCE
##########################################################


tmin_cov, tmax_cov = -0.3, 0

cov_methods = ['auto']

##########################################################
### SOURCE SPACE
##########################################################


src_spacing = 'oct6'

##########################################################
### BEM
##########################################################

# bem parameters
bem_ico = 4
conductivity_1 = [0.3]  # for single layer
conductivity_3 = [0.3, 0.006, 0.3]  # for three layers

##########################################################
### FORWARD AND INVERSE OPERATORS
##########################################################


# for inverse operator

snr = 3.
lambda2 = 1.0 / snr ** 2

snr_epoch = 3.
lambda2_epoch = 1.0 / snr_epoch ** 2

subject_to = 'fsaverage'
spacing_morph = 5

sd_categories = ['visual', 'hear', 'hand', 'emotional', 'neutral', 'pwordc',
                 'target']
ld_categories = ['visual', 'hear', 'hand', 'emotional', 'neutral', 'pwordc',
                 'pworda', 'filler']
fname_sd = {'fruit': 'block_fruit_epochs-epo.fif',
            'odour': 'block_odour_epochs-epo.fif',
            'milk': 'block_milk_epochs-epo.fif',
            'SD_words': 'block_SD_words_epochs-epo.fif'}

fname_ld = {'LD': 'block_LD_epochs-epo.fif',
            'LD_words': 'block_LD_words_epochs-epo.fif'}

targets = ['fruit', 'odour', 'milk']
# signal_mode=['EMEG','EEG','MEG']
signal_mode = ['EMEG']

inv_op_sd = {}
inv_op_ld = {}
##########################################################
### APPLY INVERSE OPERATOR TO EVOKED DATA
##########################################################


##########################################################
### Plot
##########################################################

subjects_trans = [
    # s  'MRI_0030-trans.fif',#0
    'MRI_0032-trans.fif',  # 1
    'MRI_0034-trans.fif',  # 2
    'MRI_0035-trans.fif',  # 3
    'MRI_0042-trans.fif',  # 4
    'MRI_0045-trans.fif',  # 5
    'MRI_0052-trans.fif',  # 6
    'MRI_0056-trans.fif',  # 7
    'MRI_0069-trans.fif',  # 8
    'MRI_0070-trans.fif',  # 9
    'MRI_0072-trans.fif',  # 10
    'MRI_0073-trans.fif',  # 11
    'MRI_0075-trans.fif',  # 12
    'MRI_0078-trans.fif',  # 13
    'MRI_0082-trans.fif',  # 14
    'MRI_0086-trans.fif',  # 15
    'MRI_0097-trans.fif',  # 16
    'MRI_0122-trans.fif',  # 17
    'MRI_0125-trans.fif',  # 18
]

pictures_path_Source_estimate = os.path.expanduser('~') + \
                                '/Python/pictures/Source_Signals/'
# pictures_path = os.path.expanduser('~') + '/Python/pictures/Source_Signals/'
pictures_path_evoked_white = os.path.expanduser('~') + \
                             '/Python/pictures/evoked_white/'
pictures_path_cove = os.path.expanduser('~') + '/Python/pictures/plot_cov/'
pictures_path_grand_average = os.path.expanduser('~') + \
                              '/Python/pictures/SD-LD_task_pictures/'

# stc_SD_words_all = np.zeros([20484, 901])
# stc_LD_words_all = np.zeros([20484, 901])
# stc_SD_LD_words_all = np.zeros([20484, 901])

block_names = ['fruit', 'odour', 'milk', 'SD', 'LD']

categories_sd = ['visual', 'hear', 'hand', 'emotional', 'neutral', 'pwordc',
                 'target', 'words']
categories_ld = ['visual', 'hear', 'hand', 'emotional', 'neutral', 'pwordc',
                 'pworda', 'filler', 'words']

###########################################################
# stc.plot / evoked distribution, t-maps, Cluster_based permutation test
###########################################################
time_window = [0.050, 0.150, 0.250, 0.350, 0.450]
# time_window = np.arange(0.050,0.450,0.200)

# time_window = [0.050]
con_time_window = [50, 250, 450]
baseline_time_window = [-200, 0]

# con_time_window = [0.050]

con_freq_band_psi = [4, 16, 26, 35]
con_freq_band = [4, 8, 16, 26, 35]

stc_all = []
min_max_val = []
# time_window_len = 0.100
time_window_len = 0.1

# time_window_len = 0.500

# time_window_len = 0.800

con_time_window_len = 200
sfreq = 1000
im_coh_sd = np.zeros([len(subjects), len(con_freq_band) - 1,
                      len(con_time_window), 6, 6])
im_coh_ld = np.zeros([len(subjects), len(con_freq_band) - 1,
                      len(con_time_window), 6, 6])
im_coh_sd_ld = np.zeros([len(subjects), len(con_freq_band) - 1,
                         len(con_time_window), 6, 6])
im_coh_sd_sorted = np.zeros([len(con_time_window),
                             len(con_freq_band) - 1, 6, 6])
im_coh_ld_sorted = np.zeros([len(con_time_window),
                             len(con_freq_band) - 1, 6, 6])

#  Timeseries: ROIs
epochs_names = ['block_SD_words_epochs-epo.fif',
                'block_LD_words_epochs-epo.fif']
inv_op_name = ['InvOp_SD_EMEG-inv.fif', 'InvOp_LD_EMEG-inv.fif']
pvalue = 0.05
rois_labels = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
n_permutations = 5000


###########################################################
# mthod TLMDPC semantic analysis
###########################################################

colors1 = [
    "dimgrey",
    "darkred",
    "red",
    "orange",
    "gold",
    "yellow",
    "white",
]



colors2 = ["blue", "green", "white", "yellow", "red"]
colors3 = ["darkgrey", "grey","dimgrey", "black", "dimgrey","grey", "darkgrey"]

background_color = "white"
font_color = "black"


cmap_name = "my_list"
n_bin = 100
cm1 = LinearSegmentedColormap.from_list(cmap_name, colors1, N=n_bin)
cm2 = LinearSegmentedColormap.from_list(cmap_name, colors2, N=n_bin)
cm3 = LinearSegmentedColormap.from_list(cmap_name, colors3, N=n_bin)

tail = 0
t_threshold = -stats.distributions.t.ppf(pvalue / 2.0, len(subjects) - 1)


def mask_function(X, cut_off=None):
    if cut_off is not None:
        r, c = X.shape
        for i in np.arange(0, r):
            for j in np.arange(0, c):
                if X[i, j] < cut_off:
                    X[i, j] = cut_off
    return X