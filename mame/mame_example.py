'''
Multi-Attribute Matching Example
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Example use of MAME, all parameter values from paper:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

import ffinding_module as mod
import scoring_module as smod

# Scoring weights
points = [2, # IMS-MS, high int
          1, # IMS-MS, low int
          3, # IMS-MS, low CCS error
          4, # FT-ICR, high int
          2, # FT-ICR, low int
          3, # FT-ICR, isotopic signature seen
          1, # Additional adducts
          0.5, # Additional features (per adduct)
          2, # Detected by both MS's
          4, # Unique mass
          1 # High mass
          ]

# Parameter values
params = [1000,    # 0, IMS-MS, pos min int*
          1000,    # 1, IMS-MS, neg min int*
          6,       # 2, IMS-MS, max mass error*
          3,       # 3, IMS-MS, min acceptable samples*
          1,       # 4, IMS-MS, max acceptable blanks*
          5,       # 5, IMS-MS, max CCS error**
          1,       # 6, FTICR-MS, pos min int*
          1,       # 7, FTICR-MS, neg min int*
          1.5,     # 8, FTICR-MS, max mass error*
          1,       # 9, FTICR-MS, min samples*
          0,       # 10, FTICR-MS, max blanks*
          200,     # 11, High mass cutoff**
          1,       # 12, IMS-MS, mass shift*
          0,       # 13, FTICR-MS, mass shift*
          0        # 14, 0 = score all matched features, 1 = score only if CCS error < PARAMS[5]*
          ]

# Set these parameter values to be used everywhere
mod.params = params

# Adduct information, order: ['H', 'Na', 'De', 'Cl']
adducts = [1.0078246, 22.9898, -1.0078246, 34.968853]

# Static library data
formulas, masses, ccs_vals, present, _ = mod.get_library_info()

# Read in all FTICR features then remove any points that are not within
# MAX_FTICR_ERROR ppm (huge time savings)
fticr_pos, fticr_neg, pos_formulas, neg_formulas = mod.read_fticr_data() 
fticr_pos = mod.downselect_fticr(masses, adducts, fticr_pos)
fticr_neg = mod.downselect_fticr(masses, adducts, fticr_neg)

# Read in all IMS-MS features
imsms_pos, imsms_neg = mod.gen_imsms_data()

# Match up features to library entries in each mix
all_mixes, shifts = mod.feature_matching(10, imsms_pos, imsms_neg,
                                         fticr_pos, fticr_neg, pos_formulas,
                                         neg_formulas, adducts, masses,
                                         formulas, ccs_vals)

# Summarize results for fast scoring
unique = smod.label_unique_masses(masses)  # Changes based on mass error
tallies, _ = smod.gen_all_vectors(all_mixes, masses, unique)

# Find max AUPR for this set up (sample 100k different sets of point
# values, pick best)
top_scores, top_points = smod.monte_carlo(tallies, present)
top_scores, top_points = zip(*sorted(zip(top_scores, top_points)))
points = top_points[0]
aupr = top_scores[0]

# Find optimal threshold for this set of points
_, thresh = smod.find_threshold(points, tallies, present)

# Calc stats
fnr, fdr, acc = smod.calc_stats(points, tallies, present, thresh)

# Print results
print('Results:')
print('\tFNR: %.2f' % fnr)
print('\tFDR: %.2f' % fdr)
print('\tAcc: %.2f' % acc)