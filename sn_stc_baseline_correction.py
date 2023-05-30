#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:23:19 2020

@author: sr05
"""
import numpy as np

def stc_baseline_correction(X,times):
    time_dim = len(times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    baseline_timepoints = times[0:300]

    baseline_mean = X.data[:,0:len(baseline_timepoints)].mean(1)
    # baseline_mean = X[0:len(baseline_timepoints)].mean()

    baseline_mean_mat = np.repeat(baseline_mean.reshape([len(baseline_mean),1]),\
                                  time_dim  ,axis=1)
    corrected_stc = X - baseline_mean_mat
    return corrected_stc

