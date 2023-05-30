"""
============================================================
Visualize source leakage among labels using a circular graph
============================================================
 
This example computes all-to-all pairwise leakage among 68 regions in
source space based on MNE inverse solutions and a FreeSurfer cortical
parcellation. Label-to-label leakage is estimated as the correlation among the
labels' point-spread functions (PSFs). It is visualized using a circular graph
which is ordered based on the locations of the regions in the axial plane.
"""
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#
# License: BSD (3-clause)
 
# Copied to test laterality leakage in Setareh's analysis
 
import numpy as np
import matplotlib.pyplot as plt
 
import mne
from mne.datasets import sample
from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread)
 
from mne.viz import circular_layout, plot_connectivity_circle
 
# Labels for Setareh's data
from SN_semantic_ROIs import SN_semantic_ROIs
 
print(__doc__)
 
###############################################################################
# Load forward solution and inverse operator
# ------------------------------------------
#
# We need a matching forward solution and inverse operator to compute
# resolution matrices for different methods.
 
data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-fixed-inv.fif'
forward = mne.read_forward_solution(fname_fwd)
# Convert forward solution to fixed source orientations
mne.convert_forward_solution(
    forward, surf_ori=True, force_fixed=True, copy=False)
inverse_operator = read_inverse_operator(fname_inv)
 
# Compute resolution matrices for MNE
rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                        method='MNE', lambda2=1. / 3.**2)
src = inverse_operator['src']
del forward, inverse_operator  # save memory
 
###############################################################################
# Read and organise labels for cortical parcellation
# --------------------------------------------------
#
 
# Setareh's labels
labels = SN_semantic_ROIs()
 
# morphing ROIs from fsaverage to sample
labels = mne.morph_labels(labels, subject_to='sample',
                          subject_from='fsaverage',
                          subjects_dir=data_path + '/subjects')
 
label_names = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
 
 
###############################################################################
# Compute point-spread function summaries (PCA) for all labels
# ------------------------------------------------------------
#
# We summarise the PSFs per label by their first five principal components, and
# use the first component to evaluate label-to-label leakage below.
 
# Compute first PCA component across PSFs within labels.
# Note the differences in explained variance, probably due to different
# spatial extents of labels.
n_comp = 5
stcs_psf_mne, pca_vars_mne = get_point_spread(
    rm_mne, src, labels, mode='pca', n_comp=n_comp, norm=None,
    return_pca_vars=True)
 
# get PSFs for all vertices in labels
stcs_psf_mne_all = get_point_spread(
    rm_mne, src, labels, mode=None, n_comp=1, norm=None,
    return_pca_vars=False)
 
del rm_mne
 
 
### LEAKAGE INDICES
# for first SVD components
 
# PSF for left ATL
stcL = stcs_psf_mne[0]
 
# PSF for right ATL
stcR = stcs_psf_mne[1]
 
# label for left ATL
labelL = labels[0]
 
# label for right ATL
labelR = labels[1]
 
# leaked sources from left into right ATL
stc_LtoR = stcL.in_label(labelR)
 
# leaked sources from left into right ATL
stc_RtoL = stcR.in_label(labelL)
 
# "good" leakage within labels
stc_LtoL = stcL.in_label(labelL)
stc_RtoR = stcR.in_label(labelR)
 
# leakage value from left to right ATL
# based on first SVD component
leak_LtoR = np.abs(stc_LtoR.data[:, 0]).mean()
 
# leakage value from left to right ATL
leak_RtoL = np.abs(stc_RtoL.data[:, 0]).mean()
 
# "good" leakage within ATLs
leak_LtoL = np.abs(stc_LtoL.data[:, 0]).mean()
leak_RtoR = np.abs(stc_RtoR.data[:, 0]).mean()
 
# Leakage Index for left ATL
LiL = leak_RtoL / leak_LtoL
# and for right ATL
LiR = leak_LtoR / leak_RtoR
 
print('Leakage Indices (based on 1st SVD comps) left and right ATLs: %f, %f' % (LiL, LiR))
 
### now for full PSFs
 
# PSF for left ATL
stcL = stcs_psf_mne_all[0]
 
# PSF for right ATL
stcR = stcs_psf_mne_all[1]
 
# label for left ATL
labelL = labels[0]
 
# label for right ATL
labelR = labels[1]
 
# leaked sources from left into right ATL
stc_LtoR = stcL.in_label(labelR)
 
# leaked sources from left into right ATL
stc_RtoL = stcR.in_label(labelL)
 
# "good" leakage within labels
stc_LtoL = stcL.in_label(labelL)
stc_RtoR = stcR.in_label(labelR)
 
# leakage value from left to right ATL
# based on all PSFs for all vertices in label
leak_LtoR = np.abs(stc_LtoR.data).mean()
 
# leakage value from left to right ATL
leak_RtoL = np.abs(stc_RtoL.data).mean()
 
# "good" leakage within ATLs
leak_LtoL = np.abs(stc_LtoL.data).mean()
leak_RtoR = np.abs(stc_RtoR.data).mean()
 
# Leakage Index for left ATL
LiL = leak_RtoL / leak_LtoL
# and for right ATL
LiR = leak_LtoR / leak_RtoR
 
print('Leakage Indices (based on all PSFs) left and right ATLs: %f, %f' % (LiL, LiR))
 
# ### Plotting PSFs with labels
# brain = stcL.plot(subject='sample', subjects_dir=subjects_dir, hemi='both')
# brain.add_label(labelL, color='blue')
# brain.add_label(labelR, color='crimson')