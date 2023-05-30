#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:05:03 2020

@author: sr05
"""
import numpy as np
def label_baseline_correction(X,times):
    time_dim = len(times)
    # baseline_timepoints = X.times[np.where(X.times<0)]
    baseline_timepoints = times[0:300]

    baseline_mean = X[0:len(baseline_timepoints)].mean()
    # baseline_mean = X[0:len(baseline_timepoints)].mean()

    baseline_mean_mat = np.repeat(baseline_mean,\
                                  time_dim  ,axis=0)
    corrected_label = X - baseline_mean_mat
    return corrected_label