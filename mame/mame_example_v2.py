'''
Multi-Attribute Matching Example
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Example use of MAME, all parameter values from paper:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

import numpy as np

import ffinding_module as mod
import scoring_module as smod

#%% Init   
# Original point set
#points = [2, 1, 3, 4, 2, 3, 1, 0.5, 2, 4, 1]
#threshold = 6

# Optimized point set
points = [9.047, 0.953, 1.055, 0.17, 2.218, 0.628, 9.513, 1.065, 2.253, 0.167, 0.887]
threshold = 11.23

# Open library with compound info
write_wb, sheet = mod.open_library()
masses, unique, present = smod.spreadsheet_info(sheet)

# Get info for features
#all_mixes = smod.combine_mix_info()
## NEW ##
# Adduct information
adducts = [1.0078246, 22.9898, -1.0078246, 34.968853]  # ['H', 'Na', 'De', 'Cl']

# Static library data
formulas, _, ccs_vals, present, _ = mod.get_library_info()

# Read in all FTICR features then remove any points that are not within
# MAX_FTICR_ERROR ppm (huge time savings)
fticr_pos, fticr_neg, pos_formulas, neg_formulas = mod.read_fticr_data() 
fticr_pos = mod.downselect_fticr(masses, adducts, fticr_pos)
fticr_neg = mod.downselect_fticr(masses, adducts, fticr_neg)

# Read in all IMS-MS features
imsms_pos, imsms_neg = mod.gen_imsms_data('../data/IMSMS_%s_Features.csv')

# Match up features to library entries in each mix
all_mixes, shifts = mod.feature_matching(10, imsms_pos, imsms_neg,
                                         fticr_pos, fticr_neg, pos_formulas,
                                         neg_formulas, adducts, masses,
                                         formulas, ccs_vals)

# Summarize results for fast scoring
#unique = smod.label_unique_masses(masses)  # Changes based on mass error

tallies, ccs_used = smod.gen_all_vectors(all_mixes, masses, unique)

scores = smod.score_all(points, tallies)
passed = scores >= threshold
tp_count = np.sum(present * passed)
print('TP: %i' % tp_count)
print('FP: %i' % (np.sum(passed) - tp_count))