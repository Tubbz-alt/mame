'''
Scoring and Optimization Module
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Feature scoring, analysis, and optimization methods, as described in:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

#%% Imports
import numpy as np
import random
from sklearn.metrics import precision_recall_curve, auc

#%% Scoring Functions

def extract_rows(df, col, val):
    '''
    Return subset DataFrame with the given val in col
    '''
    return df.loc[df[col] == val]

def feature_points(df, adducts):
    '''
    Sums intensities and counts number of features in DataFrame. Does
    separately for pos and neg mode.
    '''
    
    # Init
    summed_intensity_pos = 0
    addl_features_matched_pos = 0
    summed_intensity_neg = 0
    addl_features_matched_neg = 0
    
    # Cycle through adducts
    for j, adduct in adducts.iterrows():
        df_adduct = extract_rows(df, 'Adduct', adduct['Name'])
        if len(df_adduct) > 0:
            if adduct['Charge'] > 0:
                summed_intensity_pos += df_adduct['Mean Intensity'].sum(axis=0)
                addl_features_matched_pos += len(df_adduct) - 1
            elif adduct['Charge'] < 0:
                summed_intensity_neg += df_adduct['Mean Intensity'].sum(axis=0)
                addl_features_matched_neg += len(df_adduct) - 1
    
    # Take log of intensities
    if summed_intensity_pos != 0:
        summed_intensity_pos = np.log10(summed_intensity_pos)
    if summed_intensity_neg != 0:
        summed_intensity_neg = np.log10(summed_intensity_neg)

    # Prep results and return
    summed_intensity = [summed_intensity_pos, summed_intensity_neg]
    addl_features_matched = addl_features_matched_pos + addl_features_matched_neg
    return summed_intensity, addl_features_matched

# Scores features matched to sample + compound 
def gen_vector(df, config, adducts, point_num=0):
    '''
    Creates vector of all counts
    
    Order:
        - Pos mode summed intensities
        - Neg mode summed intensities
        - Number of features with low CCS error (<= config['CCSError'])
        - Number of additional adducts (total adducts - 1)
        - Number of additional features (total features - 1, counted \
          separately for each adduct then added together)
        - Summed MS2 similarity scores
        - Number of close MS2s (similarity >= config['HighMS2Thresh'])
    '''

    vector = np.zeros(point_num, dtype=np.float64)
    
    if len(df) > 0: 

        # Pos adducts
        # TODO remove hard coding adducts
        summed_intensity, addl_features_matched = feature_points(df, adducts)
        vector[0] += summed_intensity[0]
        vector[1] += summed_intensity[1]
        vector[4] += addl_features_matched  # Counted per adduct

        # Check for low CCS errors
        l = []
        [[l.append(x) for x in y] for y in df['CCS Error']]  # Make 1 list
        l = [x for x in l if np.abs(x) <= config['CCSError']]  # Remove high error
        vector[2] = len(l)

        # Additional adducts
        adducts_observed= df['Adduct'].unique()
        vector[3] = len(adducts_observed) - 1
        
        # MS2 - cycle through similarity score lists
        for index, row in df.iterrows():
            for l in row['MS2 Similarity']:
                sim_scores = np.array(l)
                vector[5] += sum(sim_scores)
                vector[6] += len(np.where(sim_scores >= config['HighMS2Thresh'])[0])

    return vector

def gen_all_vectors(matched_features, config, adducts, cpd_num=0,
                    sample_names=[], point_num=7):
    '''
    Cycle through all compounds and generate vector (with gen_vector()) of
    total evidence for each sample.
    
    Returns AxBxC matrix. A=Number of Samples, B=Number of Compounds,
    C=Length of Evidence Vector (7)
    '''
    score_matrix = np.zeros((len(sample_names), cpd_num, point_num))
    for mix in range(len(sample_names)):
        this_sample = extract_rows(matched_features, 'Sample', sample_names[mix])
        for i in range(cpd_num):
            this_sample_cpd = extract_rows(this_sample, 'Cpd No', i)  # Ensure this col is int
            score_matrix[mix, i, :] = gen_vector(this_sample_cpd, config,
                                                 adducts, point_num=point_num)
    return score_matrix

def score_all(points, tallies):
    '''
    Generate scores for all samples/compounds. Requires tallies, the
    output from gen_all_vectors().
    '''
    scores = np.zeros((tallies.shape[0], tallies.shape[1]))
    for i in range(tallies.shape[0]):
        scores[i, :] = np.dot(tallies[i, :, :], points)
    return scores.T

#%% Scoring Analysis - For Samples with Known Composition

def get_pr_re(scores, present):
    '''
    Calculate precision and recall for range of threshold values (chosen
    automatically).
    '''
    l = present.shape[0] * present.shape[1]
    scores = scores.reshape((l, 1))
    labels = present.reshape((l, 1))
    return precision_recall_curve(labels, scores, pos_label=1)


def calc_aupr(scores, present):
    '''
    Calculate area under the precision recall curve
    '''
    pr, re, th = get_pr_re(scores, present)
    
    if len(th) == 1:
        # Everything has the same score. Edge case we want to ignore.
        return 0
    
    aupr = auc(re, pr)
    return -aupr


def find_threshold(scores, present, num=2):
    '''
    Find threshold that maximizes AUPR
    '''
    pr, re, threshold = get_pr_re(scores, present)

    eq = 2 * pr * re / (pr + re)
    ind = np.where(eq == np.nanmax(eq[1:-1]))[0][0]
   
    return ind, threshold[ind]

def monte_carlo(tallies, present, track_num=10, n=1000000, point_num=0):
    '''
    Perform Monte Carlo sampling on point values to use when scoring.
    For each sample, calculates AUPR and stores the top track_num.
    '''
    top_scores = [0] * track_num
    top_points = [[0] * point_num] * track_num
    min_score = 0
    
    for i in range(1, n + 1): # Run until stopped

        # Test new set of points
        points = [random.randint(0, 1000)/100. for x in range(point_num)]
        scores = score_all(points, tallies)
        total_score = -calc_aupr(scores, present)
        
        # Add to tracked scores/points if better than the lowest currently
        # tracked
        if total_score > min_score:
            ind = top_scores.index(min_score)
            top_scores[ind] = total_score
            top_points[ind] = points
            min_score = min(top_scores)
    
    # Rate using top 10%
    return top_scores, top_points

def calc_reproducibility(points, tallies, present, thresh):
    '''
    Calculates reproducibility, a metric used by the EPA when reporting
    for their samples. Has not been checked to ensure we're calculating
    in exactly the same way.
    '''

    # Find which compounds were spiked in multiple times
    sum_present = np.sum(present, axis=1)  # Sum across columns
    ind = np.where(sum_present > 1)[0]  # Ind for those spiked into more than one mix
    sum_multispiked_present = sum_present[ind]  # Extract only multispiked counts
    
    # Find which of these were correctly identified in each mix
    scores = score_all(points, tallies)  # Use scoring system to assign conf. scores
    multispiked_scores = scores[ind, :]  # Downselect to multispiked
    multispiked_passed = multispiked_scores > thresh  # Contains TP and FP
    multispiked_present = present[ind, :]  # Find which were actually spiked in
    multispiked_tp = multispiked_present * multispiked_passed  # Now contains only TP
    sum_multispiked_tp = np.sum(multispiked_tp, axis=1, dtype=np.float)
    
    # Find % TP for compounds that were spiked in multiple times
    reproducibility = sum_multispiked_tp / sum_multispiked_present
    
    # Remove any we failed to ID across ALL mixes
    reproducibility = reproducibility[np.where(reproducibility > 0)]
    reproducibility = np.mean(reproducibility) * 100
    sensitivity = np.sum(sum_multispiked_tp) / np.sum(sum_multispiked_present) * 100

    return round(reproducibility, 1), round(sensitivity, 1)

def calc_stats(scores, present, thresh):
    '''
    Calculates false negative rate, false discovery rate, and accuracy.
    '''
    passed = scores >= thresh
    not_passed = 1 - passed
    not_present = 1 - present
    
    tp = np.sum(passed * present, axis=0, dtype=np.float64)
    fp = np.sum(not_present * passed, axis=0, dtype=np.float64)
    tn = np.sum(not_present * not_passed, axis=0, dtype=np.float64)
    fn = np.sum(present * not_passed, axis=0, dtype=np.float64)
    
    fnr = np.nan_to_num(fn / (fn + tp))
    fdr = np.nan_to_num(fp / (fp + tp))
    acc = np.nan_to_num((tp + tn) / (tp + tn + fp + fn))
    
    return [round(np.mean(x * 100), 1) for x in [fnr, fdr, acc]]
