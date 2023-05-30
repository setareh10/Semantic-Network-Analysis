#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:37:43 2020

@author: sr05
"""
import numpy as np
def matrix_mirror(X):
    r = X.copy()
    R = r.copy()
    for k in np.arange(0,18):
        for q in np.arange(0,2):
            for t in np.arange(0,2):
                for i in np.arange(0,6):
                    for j in np.arange(i,6):
    
                        # print(i,j)
                        if i==j:
                            R[k,q,t,j,j] = 1
                        else:
                            R[k,q,t,i,j] = r[k,q,t,j,i]
    return R                
                    