'''
Scoring and Optimization Module
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Feature scoring, analysis, and optimization methods as described in:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

#%% Imports
from time import time
import random

import numpy as np
from pyswarm import pso
from sklearn.metrics import precision_recall_curve, auc

from ffinding_module import PARAMS, in_list, column_string

FAX=8
FLAB=8
C1_1 = (220/255., 41/255., 12/255.)
C1_2 = (45/255., 145/255., 167/255.)
C2_1 = (99/255., 198/255., 10/255.)
C2_2 = (184/255., 156/255., 239/255.)
FIG_DPI = 500

#%% Data Prep Functions

# Scrape info needed from lib. There's better/faster ways to do this.
# To be replaced by ffinding_module's get_library_info function
def spreadsheet_info(sheet, mass_col='I', unique_col='M', present_col_num=14):
    masses = []
    unique = []
    present = []
    for i in range(2, sheet.max_row + 1):
        masses.append(sheet[mass_col + str(i)].value)
        unique.append(sheet[unique_col + str(i)].value)
        
        temp = []
        for j in range(10):
            present_col = column_string(present_col_num + j)
            temp.append(sheet[present_col + str(i)].value)
        present.append(temp)
    masses = np.array(masses, dtype=np.float32)
    unique = np.array(unique, dtype=np.float32)
    present = np.array(present, dtype=np.bool)
    
    return masses, unique, present


# For a list of masses, find how many neighbors they have within that
# list (within 6 ppm), counting themselves
def label_unique_masses(library_masses):
    unique = np.zeros(len(library_masses))
    other_masses = list(np.copy(library_masses))
    for i in range(len(library_masses)):
        this_mass = library_masses[i]
        temp_other_masses = other_masses[:]
        temp_other_masses.remove(this_mass)
        closest_neighbor = in_list(temp_other_masses, this_mass)
        while closest_neighbor is not None:
            # This mass is NOT unique within the current mass error range
            unique[i] += 1
            temp_other_masses.remove(closest_neighbor)
            closest_neighbor = in_list(temp_other_masses, this_mass)
    return unique


def combine_mix_info(name='../data/AllFeatureInfo/Mix'):
    all_mixes = np.array(np.load('%s%i.npy' % (name, 0))[1:, :], dtype=np.float32)
    col = np.array([0] * all_mixes.shape[0], ndmin=2).T
    all_mixes = np.concatenate((col, all_mixes), axis=1)
    
    for mix in range(1, 10):
        m = np.array(np.load('%s%i.npy' % (name, mix))[1:, :], dtype=np.float32)
        col = np.array([mix] * m.shape[0], ndmin=2).T
        m = np.concatenate((col, m), axis=1)
        all_mixes = np.concatenate((all_mixes, m))
        
    return all_mixes

#%% Scoring

# Helper function for the precision_recall_curve function
def get_pr_re(points, m, present):
    scores = score_all(points, m)
    l = present.shape[0] * present.shape[1]
    scores = scores.reshape((l, 1))
    labels = present.reshape((l, 1))
    return precision_recall_curve(labels, scores, pos_label=1)


# Calculate the AUPR
def calc_aupr(points, m, present):
    pr, re, th = get_pr_re(points, m, present)
    
    if len(th) == 1:
        # Everything has the same score. Edge case we want to ignore.
        return 0
    
    aupr = auc(re, pr)
    return -aupr


# Find the F1 threshold that balances precision and recall
def find_threshold(points, m, present, num=2):
    pr, re, threshold = get_pr_re(points, m, present)

    eq = 2 * pr * re / (pr + re)
    ind = np.where(eq == np.nanmax(eq[1:-1]))[0][0]
   
    return ind, threshold[ind]


# Scores the confidence of all compounds given their evidence and a set
# of scoring weights
def score_all(points, m):
    m = np.copy(m)
    scores = np.zeros((m.shape[0], m.shape[1]))
    for i in range(10):
        scores[i, :] = np.dot(m[i, :, :], points)
    return scores.T


# Gather all evidence for each suspect and place the counts in one
# large matrix
def gen_all_vectors(all_mixes, masses, unique):
    
    mix_num = 10
    compound_num = masses.shape[0]
    point_num = 11
    score_matrix = np.zeros((mix_num, compound_num, point_num))
    ccs_used = np.zeros((mix_num, compound_num))
    for mix in range(mix_num):

        m = all_mixes[np.where(all_mixes[:, 0] == mix)[0], 1:]
        
        for i in range(compound_num):
            
            feature_info = np.array(m[np.where(m[:, 0] == float(i + 2))[0], 1:])
            temp = gen_vector(feature_info, unique[i], masses[i])
            score_matrix[mix, i] = temp
            ccs_used[mix, i] = temp[2] > 0

    return score_matrix, ccs_used.T


# Gathers all evidence for a single compound. Places counts into one
# vector
def gen_vector(feature_info, masses_in_range, mass):

    passing_feat = False
    vector = np.zeros(11)
    
    if feature_info.shape[0] > 0: 

        # FTICR Scoring
        info = feature_info[np.where(feature_info[:, 0] == 0)[0], 1:]
        if info.shape[0] > 0:
            passing_feat = True
            temp = instrument_points(info, PARAMS[11], PARAMS[12])
            vector[3:5] = temp[0:2]
            vector[6:8] += temp[2:]
            
            # Check for iso sig
            vector[5] = info[0, -1]
        
        # IMSMS Scoring
        info = feature_info[np.where(feature_info[:, 0] == 1)[0], 1:]
        
        if info.shape[0] > 0:
            passing_feat = True
            temp = instrument_points(info, PARAMS[5], PARAMS[6])
            vector[0:2] = temp[0:2] # High/low int numbers for IMS MS
            vector[6:8] += temp[2:] # Add to addl adducts and features
            
            # Check for low CCS errors
            vector[2] = len([x for x in info[:, -1] if abs(x) <= PARAMS[4]])
    
        # Other
        if passing_feat:
            # Detected by both MS's
            if (0 in feature_info[:, 0]) and (1 in feature_info[:, 0]):
                vector[8] = 1
                        
            # High mass
            if mass > PARAMS[13]:
                vector[10] = 1
    
            # Unique mass
            if masses_in_range == 1:
                vector[9] = 1
            else: # Divide by total number of mass candidates
                vector /= float(masses_in_range)
                vector[2] *= masses_in_range # Remultiply CCS

    return vector


# Count up evidence associated with instruments (IMSMS and FTICR)
def instrument_points(info, pos_cutoff, neg_cutoff):
    vector = np.zeros(4)
    
    # Pos adducts - count high and low intensity
    vector[:2] += pos_neg_feature_points(info, [0, 1], pos_cutoff)
    
    # Neg adducts - count high and low intensity
    vector[:2] += pos_neg_feature_points(info, [2, 3], neg_cutoff)
    
    # Additional adducts
    addl_adducts = len(list(set(info[:, 0]))) - 1
    vector[2] = addl_adducts
    
    # Additional features
    vector[3] = count_addl_features(info)
    
    return vector


# Helper function for instrument_points()
def pos_neg_feature_points(info, l, cutoff):
    vector = np.zeros(2)
    ind = [i for i in range(info.shape[0]) if info[i, 0] in l]
    adducts = info[ind, :]
    vector[0] = len([x for x in adducts[:, 2] if x > cutoff])
    vector[1] = len(ind) - vector[0]
    return vector


# Count how many features (beyond the first one) there are for a given
# adduct/mode pair
def count_addl_features(info):
    addl_features = 0
    adduct_nums = list(set(info[:, 0]))
    for num in adduct_nums:
        addl_features += len([x for x in info[:, 0] if x == num]) - 1
    return addl_features


#%% Score Optimization

# Monte Carlo Optimization - randomly initialize point values and test
# how well they do
def monte_carlo(tallies, present, track_num=10, point_num=11, n=100000):
    top_scores = [0] * track_num
    top_points = [[0] * point_num] * track_num
    min_score = 0
    
    for i in range(1, n + 1): # Run until stopped

        # Test new set of points
        points = [random.randint(0, 10000)/1000. for x in range(11)]
        total_score = -calc_aupr(points, tallies, present)
        
        # Add to tracked scores/points if better than the lowest currently
        # tracked
        if total_score > min_score:
            ind = top_scores.index(min_score)
            top_scores[ind] = total_score
            top_points[ind] = points
            min_score = min(top_scores)
    
    # Rate using top 10%
    return top_scores, top_points


# PSO optimization via PySwarm - sample across all possible point-value
# space and use results to find a minima
def score_pyswarm(tallies, present):
    t = time()

#    PARAMS[8] = 0.5
    
    lb = [0] * 11
    ub = [10] * 11
    points, aupr = pso(calc_aupr, lb, ub, args=(tallies, present), swarmsize=3000, 
                       maxiter=100, minstep=0.001)#, omega=-0.2134, phip=-0.3344, phig=2.3259)
    
    print('Time taken: %.1f min' % ((time() - t) / 60))

    return list(points), aupr 


#%% Scoring Analysis

# Calculate FNR, FDR, and accuracy
def calc_stats(points, tallies, present, thresh):
    scores = score_all(points, tallies)  # Use scoring system to assign conf. scores
    passed = scores >= thresh
    not_passed = 1 - passed
    not_present = 1 - present
    
    tp = np.sum(passed * present, axis=0, dtype=np.float64)
    fp = np.sum(not_present * passed, axis=0, dtype=np.float64)
    tn = np.sum(not_present * not_passed, axis=0, dtype=np.float64)
    fn = np.sum(present * not_passed, axis=0, dtype=np.float64)
    
    fnr = fn / (fn + tp)
    fdr = fp / (fp + tp)
    acc = (tp + tn) / (tp + tn + fp + fn)
        
    return [round(np.mean(x * 100), 1) for x in [fnr, fdr, acc]]
