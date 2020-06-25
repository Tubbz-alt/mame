# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:27:46 2020

@author: MontyPypi
"""

from bisect import bisect_left
from collections import defaultdict as ddict
import math
import numpy as np

import formula_module as fmod

#%% Misc. Helper Functions
def column_string(n):
    '''
    Converts numbers to their corresponding column name in Excel.
    
    Parameters
    ----------
    n: int
        Column number
    
    Returns
    ----------
    s: string
        Excel column string
        
    Examples
    ----------
    >>> column_string(5)
    'E'
    
    >>> column_string(31)
    'AE'
    '''
    
    div = n
    s = ''
    while div > 0:
        module = (div - 1) % 26
        s = chr(65 + module) + s
        div=int((div - module) / 26)
    return s


def calc_error(val1, val2, absolute=False):
    '''
    Calculates the error between two values.
    Equation used: (val2 - val1) / val2
    
    Parameters
    ----------
    val1: float
        First number, taken to be the "new" value.
    val2: float
        Second number, taken to be the "true" or "accepted" value
    absolute: boolean (optional)
        If true, returns the absolute value of the error. Default value is False.
        
    Returns
    ----------
    e: float
        Error between val1 and val2
    '''
    if val1 is None or val2 is None:
        return None
    if absolute:
        return abs(val1 - val2) / val2
    else:
        return (val1 - val2) / val2


def ppm_error(val1, val2, absolute=False):
    '''
    Calculates the error (in ppm) between two values.
    Equation used: (val2 - val1) / val2 * 1e6
    
    Parameters
    ----------
    val1: float
        First number, taken to be the "new" value.
    val2: float
        Second number, taken to be the "true" or "accepted" value
    absolute: boolean (optional)
        If true, returns the absolute value of the error. Default value is False.
        
    Returns
    ----------
    e: float
        Error between val1 and val2 in ppm
    '''
    return calc_error(val1, val2, absolute=absolute) * 1e6


def percent_error(val1, val2, absolute=False):
    '''
    Calculates the error (in percent) between two values.
    Equation used: (val2 - val1) / val2 * 1e2
    
    Parameters
    ----------
    val1: float
        First number, taken to be the "new" value.
    val2: float
        Second number, taken to be the "true" or "accepted" value
    absolute: boolean (optional)
        If true, returns the absolute value of the error. Default value is False.
        
    Returns
    ----------
    e: float
        Error between val1 and val2 in percent
    '''
    return calc_error(val1, val2, absolute=absolute) * 1e2


def find_closest_sorted(l, value):
    '''
    Within a sorted (smallest to largest) list, l, finds the number
    closest to a given value.
    If two numbers are equally close, returns the smallest number.
    '''
    pos = bisect_left(l, value)
    if pos == 0:
        return l[0]
    if pos == len(l):
        return l[-1]
    before = l[pos - 1]
    after = l[pos]
    if after - value < value - before:
       return after
    else:
       return before


def find_closest(l, val):
    '''
    Within a list, l, finds the number closest to a given value.
    If two numbers are equally close, returns the number that comes
    first in the list.
    '''
    return min(l, key=lambda x:abs(x-val))


def in_list(l, val, e, sorted_list=False, a=False):
    '''
    Checks if an element is within a given list within a given error.
    If it is found to be in the given list, returns the value closest
    to it.
    
    Parameters
    ----------
    l: list
        Elements to search. Can be sorted or unsorted.
    val: float
        Value to search for within list
    e: float
        Allowable error to count val as equivalent to an element in the
        list.
    sorted_list: boolean (optional)
        Pass in True if the list is sorted.
        Can make this function significantly faster with large lists.
        
    Returns
    ----------
    res: number, same type as elements in given list
        Closest element within the list if it is with e of the given
        val.
        If no elements are close enough to val, returns None.
    '''
    
    # Exit for invalid list
    if len(l) == 0:
        return None
        
    # Find closest mass in list
    if sorted_list:  # Achieve speed up using binary search
        closest = find_closest_sorted(l, val)
    else:
        closest = find_closest(l, val)
        
#    if not a and np.isclose(closest, val, rtol=e): # THIS CHANGED from divide by 1mill
    if not a and calc_error(closest, val, absolute=True) < e:
        return closest
    if a and np.isclose(closest, val, atol=e):  # Need to change this
        return closest
    return None

def label_unique_masses(library_masses, e):
    '''
    Counts all masses within +/- e ppm of each mass in the list. For example,
    if 3 other masses in the given list are within error, returns 4 at that
    location in the list (since it includes the given mass as well).
    '''
    unique = np.zeros(len(library_masses))
    other_masses = list(np.copy(library_masses))
    for i in range(len(library_masses)):
        this_mass = library_masses[i]
        temp_other_masses = other_masses[:]
        temp_other_masses.remove(this_mass)
        closest_neighbor = in_list(temp_other_masses, this_mass, e)
        while closest_neighbor is not None:
            # This mass is NOT unique within the current mass error range
            unique[i] += 1
            temp_other_masses.remove(closest_neighbor)
            closest_neighbor = in_list(temp_other_masses, this_mass, e)
    return unique + 1

# Adduct should be a df with rows "Name", "M", "Charge", "Added",
# and "Removed"
def calc_new_mz(formula, adduct):
    '''
    Calculates the mz of the given formula + adduct. Note: does ensure
    the final formula of the adduct is valid (e.g. can not remove H from
    a compound that does not contain it), but otherwise assumes the
    Added/Removed components of the adduct are chemically possible.
    
    Parameters
    ----------
    formula: string
        Formula of the parent molecule (should have a charge of zero)
    adduct: Pandas Series
        Adduct information. Include:
            - Added: formula to be added to parent molecule
            - Removed: formula to be removed from parent molecule
            - M: number of members of the parent molecule
            - Charge: charge of the adduct
    
    
    Returns
    ---------
    mz: float
        Mass of parent + adduct. If not possible (such as removing a H
        from a compound that contains no H), returns None.
    
    Example Input
    ---------
    Parent compound formula: C6H12O6
    Adduct: [2M+Na-2H]-
        Added: Na
        Removed: H2 (or 2H)
        M: 2
        Charge: -1
    '''
                
    # Create new formula (if possible)
    formula = fmod.formula_split(formula)
    for ele in formula.keys():
        formula[ele] *= int(adduct['M'])
    formula = fmod.add_formula(formula, adduct['Added'])
    formula = fmod.remove_formula(formula, adduct['Removed'])
    
    # If adduct not possible, or leaves no atoms, return appropriately
    if formula is None:
        return None
    if len(formula) == 0:
        return 0

    # Return mass of new formula, accounting for charge
    return fmod.calculate_mass(formula) / abs(adduct['Charge'])

#%% MS2

# Assumes these lists are the same length
def convert_ms2_format(masses, ints):
    '''
    Converts lists of masses and intensities to a list of ['mass
    intensity']
    '''
    s = []
    for m, i in zip(masses, ints):
        s.append(str(m) + ' ' + str(i))
    return s
    
def non_nan_present(a):
    '''
    Checks if a real element (non-null) is available in the given list.
    '''
    nan_elements = np.sum(np.isnan(np.array(a)))
    all_elements = a.shape[0]
    for x in a.shape[1:]:
        all_elements *= x
    return nan_elements != all_elements

def scale_intensity(i):
    '''
    Scales list of intensities by dividing each by the max intensity.
    '''
    maxi = max(i)
    return [x / maxi for x in i]

def _parse_text_spectra(lines, delimiter=' ', minval=0, n=None):
    '''
    Helper function for parsing MS2 spectra (for CFM-ID output)
    '''
    
    if len(lines) == 0:
        return None, None
    
    m = []
    i = []
    
    for line in lines:
        t = line.split(delimiter)
        if len(t) > 1 and float(t[1]) >= minval:
            m.append(float(t[0]))
            i.append(float(t[1]))
    if len(m) == 0 or len(i) == 0:
        return None, None
    
    # Keep top n values
    if n is not None:
        i, m = zip(*sorted(zip(i, m)))
        i = i[-n:]
        m = m[-n:]
    
    # Scale intensities to be 0 to 1
    i = scale_intensity(i)
    
    return list(m), list(i)

def parse_text_spectra(s, delimiter=' ', newline='\n'):
    '''
    Returns a tuple for 10V, 20V, and 40V. Each tuple is of the form
    (masses, intensities). Written for CFM-ID output.
    '''
    try:
        if np.isnan(s) or s is None or 'energy0' not in s:
            return None, None, None
    except TypeError:
        pass
    
    spectra = s.replace('_x000D_', '').replace('\r', '').split(newline)

    ind10 = spectra.index('energy0')
    ind20 = spectra.index('energy1')
    ind40 = spectra.index('energy2')
    spectra_10V = _parse_text_spectra(spectra[ind10 + 1: ind20])
    spectra_20V = _parse_text_spectra(spectra[ind20 + 1: ind40])
    spectra_40V = _parse_text_spectra(spectra[ind40 + 1:])

    return spectra_10V, spectra_20V, spectra_40V

# c and d for weighting
def cosine_similarity(a, b, c=None, d=None):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return float(cos)

## Takes in two spectra (mass and intensity for each)
## m1 is expected, m2 is experimental
## Intensities should be pre-normalized to have a sum of 100
## Error is set high, assuming a close peak wouldn't be high int if a FP
#def score_msms_similarity(m1, i1, m2, i2, e=100):
#    
#    if m1 is None or i1 is None or m2 is None or i2 is None:
#        return 0
#    
#    score = 0
#    sum_int = sum(i1)
#    for m, i in zip(m1, i1):
#        
#        # Find median intensity of all features within MASS_ERROR
#        try:
#            errors = [ppm_error(x, m, absolute=True) for x in m2]
#        except TypeError:
#            pass
#        intensities = []
#        for j in range(len(errors)):
#            if errors[j] <= e:
#                intensities.append(i2[j])
#        if len(intensities) > 0:
#            s = cosine_similarity([0, i], [0, np.nanmedian(intensities)])
#            s *= i / sum_int * 100  # Normalize with intensity
#            score += s
#    return round(score * 10)  # for score out of 1000

# Borrowed and modified from PyMZML (github.com/pymzml/pymzML/) --spec.py
def score_msms_similarity(mz1, i1, mz2, i2, round_precision=0):
    '''
    Compares two spectra and returns cosine
    Arguments:
        spec2 (Spectrum): another pymzml spectrum that is compared to the
            current spectrum.
    Keyword Arguments:
        round_precision (int): precision mzs are rounded to, i.e. round( mz,
            round_precision ) >> 100 ppm =  1e4 = 4
    Returns:
        cosine (float): value between 0 and 1, i.e. the cosine between the
            two spectra.
    Note:
        Spectra data is transformed into an n-dimensional vector,
        where m/z values are binned in bins of 10 m/z and the intensities
        are added up. Then the cosine is calculated between those two
        vectors. The more similar the specs are, the closer the value is
        to 1.
    '''

    try:
        i1 = scale_intensity(i1)
        i2 = scale_intensity(i2)
    except ValueError:
        pass

    vector1, vector2 = ddict(int), ddict(int)
    mzs = set()
    for mz, i in zip(mz1, i1):
        vector1[round(mz, round_precision)] += i
        mzs.add(round(mz, round_precision))
    for mz, i in zip(mz2, i2):
        vector2[round(mz, round_precision)] += i
        mzs.add(round(mz, round_precision))

    z, n_v1, n_v2 = 0, 0, 0

    for mz in mzs:
        int1 = vector1[mz]
        int2 = vector2[mz]
        z += int1 * int2
        n_v1 += int1 * int1
        n_v2 += int2 * int2
    try:
        cosine = z / (math.sqrt(n_v1) * math.sqrt(n_v2))
    except:
        cosine = 0.0
    return round(cosine * 1000)