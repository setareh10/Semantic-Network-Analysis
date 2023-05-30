#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:09:08 2020

@author: sr05

=========================================================
Submit sbatch jobs for SN Connectivity
analysis
=========================================================
"""

import subprocess
from os import path as op

from importlib import reload

# import study parameters
import sn_config as C
import numpy as np

reload(C)

print(__doc__)

# # wrapper to run python script via qsub. Python3
fname_wrap = op.join('/', 'home', 'sr05',
                     'semnet-project', 'Python2SLURM.sh')
# indices of subjects to process
repeats = np.arange(100,1500,100)
# sbj_id = range(0,18)
# sbj_id = np.array([7, 11, 17])

job_list = [
    {'N': 'bv',  # job name
     'Py': 'method-monte-carlo-simulations',# Python script
     # 'ss': sbj_id,  # subject indices
     'ss': repeats,  # subject indices

     'mem': '112G',  # memory for qsub process
     'dep': ''}]

# directory where python scripts are
dir_py = op.join('/', 'home', 'sr05', 'semnet-project')

# directory for qsub output
dir_sbatch = op.join('/', 'home', 'sr05', 'semnet-project', 'sbatch_out')

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:
    for s in job['ss']:

        # Ss = str(C.subjects[s][1:-8])  # turn into string for filenames etc.
        Ss = str(s)  # turn into string for filenames etc.

        # print(Ss)

        N =  Ss + '_' + job['N']  # add number to front
        Py = op.join(dir_py, job['Py'])
        Cf = ''  # config file not necessary for FPVS
        mem = job['mem']

        # files for qsub output
        file_out = op.join(dir_sbatch, N + '.out')
        file_err = op.join(dir_sbatch, N + '.err')


        sbatch_cmd = f'sbatch -o {file_out} -e {file_err} ' \
                     f'--export=pycmd="{Py}.py {Cf}",subj_idx={Ss}, ' \
                     f'--mem={mem} --time 5-12:00:00  -J {N}  {fname_wrap} ' \
                     f'--mincpus=28 --mem-per-cpu=4G' #\
                     # f'--nodelist={nodename}'    
        # format string for display
        print_str = sbatch_cmd.replace(' ' * 25, '  ')
        print('\n%s\n' % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE,
                                shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        # Job_IDs[N, Ss] = str(int(out.split()[-1]))
