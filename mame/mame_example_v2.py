'''
Multi-Attribute Matching Example
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Example use of MAME, all parameter values from paper:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

import numpy as np

from ffinding_module import open_library
import scoring_module as smod

#%% Init   
# Original point set
#points = [2, 1, 3, 4, 2, 3, 1, 0.5, 2, 4, 1]
#threshold = 6

# Optimized point set
points = [9.047, 0.953, 1.055, 0.17, 2.218, 0.628, 9.513, 1.065, 2.253, 0.167, 0.887]
threshold = 11.23

# Open library with compound info
write_wb, sheet = open_library()
masses, unique, present = smod.spreadsheet_info(sheet)

# Get info for features
all_mixes = smod.combine_mix_info()

tallies, ccs_used = smod.gen_all_vectors(all_mixes, masses, unique)

scores = smod.score_all(points, tallies)
passed = scores >= threshold
tp_count = np.sum(present * passed)
print('TP: %i' % tp_count)
print('FP: %i' % (np.sum(passed) - tp_count))