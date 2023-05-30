#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:49:40 2022

@author: sr05
"""
lb = ['l_ATL', 'r_ATL', 'PTC', 'IFG', 'AG', 'PVA']
d_100 = np.zeros([5,6])

for n in range(0,6):
    for m in range(0,5): 
        i=18
        lATL_SD = X1[:, n, 350+100*m:350+100*(m+1)]
        lATL_LD = X2[:, n, 350+100*m:350+100*(m+1)]
        
        lATL_sd = lATL_SD.mean(1)
        lATL_ld = lATL_LD.mean(1)
    
        
     
        sd_avg = lATL_sd.mean()
        ld_avg = lATL_ld.mean()
        
        sd_var = lATL_sd.var()
        ld_var = lATL_ld.var()
        
        s = ((i-1)*(sd_var + ld_var)/(2*i))**0.5
        d_100[m,n] = (sd_avg - ld_avg)/s
        # print(lb[n], d)
        
    # plt.figure()
    # plt.plot(np.arange(1,18) , d[1:])
    # # plt.plot(np.arange(0,18) , np.ones(18)*0.5,'r')
    # # plt.plot(np.arange(0,18) , np.ones(18)*0.3,'g')

    # plt.title(lb[n])
    # plt.xlabel('sample size')
    # plt.ylabel('effect size')

    
# plt.close('all')