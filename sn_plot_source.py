#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:34:31 2020

@author: sr05
"""

import mne
import numpy as np
import sn_config as C
import matplotlib.pyplot as plt
import os

# path to filtered raw data
main_path = C.main_path
data_path = C.data_path
pictures_path = C.pictures_path_Source_estimate

# subjects' directories
subjects =  C.subjects

for n in np.arange(0,len(C.signal_mode)):
    # stc_SD_words_all = C.stc_SD_words_all
    # stc_LD_words_all = C.stc_LD_words_all
    # stc_SD_LD_words_all = C.stc_SD_LD_words_all
    
    for i in np.arange(0, len(subjects)):

    
        print('participant : ' , i , C.signal_mode[n])
        subject_from = subjects[i]
        meg = subjects[i]

    
        fname_SD_words_fsaverage = C.data_path + meg + 'block_SD_words_'+\
                                   C.signal_mode[n]+'_fsaverage'
        fname_LD_words_fsaverage = C.data_path + meg + 'block_LD_words_'+\
                                   C.signal_mode[n]+'_fsaverage'    
        stc_SD_words = mne.read_source_estimate(fname_SD_words_fsaverage )
        stc_LD_words = mne.read_source_estimate(fname_LD_words_fsaverage )

        if i==0:            
            stc_SD_words_all = stc_SD_words        
            stc_LD_words_all = stc_LD_words
        
            stc_SD_LD_words = stc_SD_words - stc_LD_words 
            stc_SD_LD_words_all = stc_SD_LD_words
        else:
                        
            stc_SD_words_all = stc_SD_words_all + stc_SD_words        
            stc_LD_words_all = stc_LD_words_all + stc_LD_words
        
            stc_SD_LD_words = stc_SD_words - stc_LD_words 
            stc_SD_LD_words_all = stc_SD_LD_words_all + stc_SD_LD_words
            
        # fig, axs = plt.subplots(3)
        # fig.suptitle('Participant : '+ meg[1:11] +' - SD_Words/LD_Words/SD-LD'\
        #              + '('+C.signal_mode[n]+')')
        # axs[0].plot(1e3 * stc_SD_words.times, stc_SD_words.data[::100, :].T)
        # plt.xlabel('time (ms)')
        # plt.ylabel(' MNE' )
        
        # axs[1].plot(1e3 * stc_LD_words.times, stc_LD_words.data[::100, :].T)
        # plt.xlabel('time (ms)')
        # plt.ylabel(' MNE' )
        
        # axs[2].plot(1e3 * stc_SD_LD_words.times, stc_SD_LD_words.data[::100, :].T)
        # plt.xlabel('time (ms)')
        # plt.ylabel('MNE' )
        # plt.savefig(pictures_path + 'Participant_' +meg[1:11] +\
        #             '_Source_estimate'+'_'+C.signal_mode[n]) 
        # plt.close('all')


    fig, axs = plt.subplots(3)
    fig.suptitle('Grand Average - SD_Words/LD_Words/SD-LD')
    axs[0].plot(1e3 * stc_SD_words.times, stc_SD_words_all.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('%s value MNE' )
    
    axs[1].plot(1e3 * stc_LD_words.times, stc_LD_words_all.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('%s value MNE' )
    
    axs[2].plot(1e3 * stc_LD_words.times, stc_SD_LD_words_all.data[::100, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('MNE')
    plt.savefig(pictures_path + 'Grand_Average-Source_estimate'+'-'+\
        C.signal_mode[n]) 
    plt.close('all')