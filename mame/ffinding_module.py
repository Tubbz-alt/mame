'''
Feature Finding Module
@author: Jamie Nunez
(C) 2018 - Pacific Northwest National Laboratory

Data processing and feature downselection methods, as described in:
Advancing Standards-Free Methods for the Identification of Small Molecules in Complex Samples
Nunez et al.
'''

#%% Imports
import numpy as np
import pandas as pd

import mame_utils

# Internal settings and column names
DEIMOS = True
FEATS_COL = 'feats'
BLANKS_COL = 'blanks'
ID_COL = 'feature_idx'
MZ_COL = 'mz'
CCS_COL = 'ccs'
RT_COL = 'rt'
MEAN_INT_COL = 'mean'
FEATURE_COLS = ['Sample', 'Cpd No', 'Adduct', 'Feature ID', 'Mean Intensity',
                'Mass Error', 'CCS Error', 'MS2 Similarity']
pd.options.mode.chained_assignment = None  # default='warn'

#%% MAME Class

class FeatureFinder:

    def _init_(self):
        return 0

    # TODO: needs to be redone with an actual reader
    def _read_config(self, filename):
        '''
        Very simple/strict configuration file reader. Refer to example config
        file for guidence. Any missing settings will be replaced with their
        default values. Each Setting (listed in set_config()) must be separated
        from its value with a " = ", and provded on separate lines. Lists must
        be provided in the format [item1, item2, item3].
        
        Parameters
        ----------
        filename: string
            Path to file to parse.
        
        Returns
        ----------
        config: dict
            Dictionary of Settings parsed from config file.
            
        Example File Contents
        ----------
        MassError = 6
        
        CCS = 2
        
        BlankIndicator = blank
        '''
        config = {}
        with open(filename) as f:
            for line in f:
                key, value = line.replace('\n', '').split(' = ')
                
                # Handle lists
                if '[' in value:
                    temp = value.replace('[', '').replace(']', '').split(', ')
                    try:
                        value = [float(x) for x in temp]
                    except ValueError:  # Must be adducts
                        value = value[1:-1].split(', ')
                
                # Handle bool
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                
                # Handle None 
                elif value.lower() == 'none':
                    value = None
                
                # Assume float or string otherwise
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Leave as string

                config[key] = value

        return config

    def set_config(self, config):
        '''
        Set available configuration parameters.
        
        Parameters
        ----------
        config: dict or string
            Dictionary to use with keys matching available Setting options
            (see below). If a string, assumed to be a path to a configuration
            file to be parsed for information. See example configuration file
            and/or read_config() for guidence.
        
        Returns
        ----------
        config: dictionary
            Final config dictionary with all settings being used moving forward.
            
        Settings & Defaults
        ----------
        Points: list of floats
            Points/multipliers assigned to matched feature aspects. Currently:
            summed intensities (positive mode), summed intensities
            (negative mode), number of CCS hits within CCSError, number of
            additional adducts (total number of adducts - 1), number of
            additional features (total number of features - 1), summed MS2
            similaroty scores, number of MS2 similarity scores > HighMS2Thresh.
            Default = [0, 0, 0, 0, 0, 0, 0]
        Adducts: list of strings
            Adducts to be searched for. Must use adduct names provided in the
            adducts.csv files, otherwise ignored. Default = ['[M+H]', '[M-H]',
            '[M+Na]'].
        MinInt: float
            Minimum intensity for a feature to be considered present.
            Default = 1.
        PosModeMinInt: float
            Minimum intensity for a feature to be considered present in
            positive mode samples. Default = MinInt.
        NegModeMinInt: float
            Minimum intensity for a feature to be considered present in
            negative mode samples. Default = MinInt.
        MinimumSamples: int
            Minimum number of replicates a feature must be observed in to be
            considered in feature matching for a given sample. Default = 1.
        MaximumBlanks: int
            Maximum number of blanks a feature can be observed and still
            considered in feature matching. Ignored if no blanks
            are available in the given data. Assumed only one set of blanks
            included for all samples in the data set (not per-sample).
            Default = 0. 
        BlankIndicator: string or None
            Substring in filenames of blank samples. If None, blanks are not
            considered. Default = None.
        RequireCCSMatch: bool
            Whether a feature must match by both mass and CCS to be considered
            a match. If False, only a mass by match is required.
            Default = False.
        MassError: float
            Maximum mass error allowed for a feature to be considered a match
            by mass. Features are not matched unless they are (at minimum) a
            match by mass. Units: ppm. Default = 6.
        CCSError: float
            Maximum CCS error allowed for a feature to be considered a match by
            CCS. If RequireCCSMatch is set to True, features are not matched
            unless they are a match by CCS as well as mass. Units: %.
            Default = 2.
        HighMS2Thresh: float
            Minimum similarity score required to be considered a match by MS2.
            Default = 800.
        Lib_MS2_ValueSplit: string
            Substring that splits mass and intensity values for MS2 in the
            library. Default = ','.
        Lib_MS2_LineSplit: string
            Substring that splits mass/intensity pairs for MS2 in the
            library. Default = ';'.
        Lib_ListSplit: string
            Substring that splits separate values provided in a single cell in the
            library. Default = '|'.
        MS2_10V_Indicator: string or None
            Substring in filenames of samples run with collision energy = 10eV.
            If None, 10 V files are not onsidered. Default = None.
            
        '''

        # Filename passed instead of dictionary
        if type(config) == str:
            config = self._read_config(config)
        
        defaults = {'Points' : [0, 0, 0, 0, 0, 0, 0],
                    'Adducts' : ['[M+H]', '[M-H]', '[M+Na]'], 'MinInt': 1 ,
                    'MinimumSamples' : 1, 'MaximumBlanks' : 0,
                    'BlankIndicator' : None, 'RequireCCSMatch' : False,
                    'MassError' : 6, 'CCSError' : 2, 'HighMS2Thresh' : 800,
                    'Lib_MS2_ValueSplit' : ',', 'Lib_MS2_LineSplit' : ';',
                    'Lib_ListSplit' : '|', 'MS2_10V_Indicator' : None,
                    'MS2_20V_Indicator' : None,
                    'MS2_40V_Indicator' : None}
        
        for key in defaults:
            if key not in config.keys():
                config[key] = defaults[key]

        if 'PosModeMinInt' not in config.keys():
            config['PosModeMinInt'] = int(config['MinInt'])

        if 'NegModeMinInt' not in config.keys():
            config['NegModeMinInt'] = int(config['MinInt'])
        
        # Set adducts using names listed in config (must match an entry in
        # adducts.csv)
        self.set_adducts(config['Adducts'])
        
        # Save config
        self.config = config

        return self.config

    def gen_report(self):
        
        #TODO: create file with time/date in name
        #      add in config settings,
        #      add any results generated by feature finding

        return

    def _process_list_cols(self, df, numeric=False):
        '''
        Returns a list of values given in a single cell.
        
        Parameters
        ----------
        df: DataFrame
            DataFrame where each column needs values split in lists.
        numeric: bool (optional)
            Whether list contents should be numeric values. If true, casted to
            float as they are split. Default: False.
        
        Returns
        ----------
        new_df: Dataframe
            DataFrame with each cell now split into a list.
        '''
        new_df = df.copy()
        new_df = new_df.applymap(str)

        list_split = self.config['Lib_ListSplit']
        if numeric:
            for col in new_df:
                new_df[col] = [[float(val) for val in s.split(list_split)] for s in new_df[col]]
        else:
            for col in new_df:
                new_df[col] = [s.split(list_split) for s in new_df[col]]

        return new_df

    # Scrape info needed from lib
    # TODO: for large mem, do not open lib directly, read in line by line in
    # boss thread, distribute to workers, return when score/results complete.
    # Will have to do softmax step after.
    def set_library(self, fname, sheet_name, header=0, formula_col='Formula',
                    mass_col='Exact Mass', ccs_indic='CCS', ms2_indic='MS/MS',
                    source_indic='Source'):
        '''
        Set the library to be used for feature matching. Creates internal
        variables (lib_formulas, lib_masses, lib_ccs, lib_msms), which can be
        accessed with get_* functions (e.g. get_lib_masses).
        
        Parameters
        ----------
        fname: string
            Path to Excel spreadsheet with library contents
        sheet_name: string
            Name of sheet containing library
        header: int (optional)
            Line number to use as the header. Zero-based index. All lines
            before will be ignored. Default = 0.
        formula_col: string (optional)
            Name of column containing compound formulas. Default = 'Formula'.
        mass_col: string (optional)
            Name of column containing compound masses. Default = 'Exact Mass'.
        ccs_indic: string (optional)
            Substring of column names that contain CCS values. Default = 'CCS'.
        ms2_indic: string (optional)
            Substring of column names that contain MS2 values. Default = 'MS/MS'
        source_indic: string (optional)
            Substring of column names that contain source values. These will be
            removed from consideration for CCS and MS2 values. Default = 'Source'

        Returns
        ----------
        pass: int
            0 is passed, 1 otherwise.
        '''
        df = pd.read_excel(fname, sheet_name=sheet_name, header=header)
        self.lib_formulas = df[formula_col].values
        self.lib_masses = df[mass_col].values  # Not actually needed
        
        # Get CCS
        cols = [x for x in df.columns if ccs_indic in x and source_indic not in x]
        self.lib_ccs = self._process_list_cols(df[cols], numeric=True)
        self.lib_ccs.reset_index(drop=True, inplace=True)

        # Get MS/MS
        cols = [x for x in df.columns if ms2_indic in x and source_indic not in x]
        self.lib_msms = self._process_list_cols(df[cols])
        self.lib_msms.reset_index(drop=True, inplace=True)
        return 0

    def get_config(self):
        '''
        Returns configuraton setting dictionary created during set_config().
        '''
        return self.config

    def get_lib_formulas(self):
        '''
        Returns library formulas pulled during set_library().
        '''
        return self.lib_formulas
    
    def get_lib_masses(self):
        '''
        Returns library masses pulled during set_library().
        '''
        return self.lib_masses

    def get_lib_ccs(self):
        '''
        Returns library CCS pulled during set_library().
        '''
        return self.lib_ccs

    def get_lib_msms(self):
        '''
        Returns library MS2 pulled during set_library().
        '''
        return self.lib_msms
    
    def get_lib_num(self):
        '''
        Returns number of compounds in library.
        '''
        return len(self.lib_formulas)
    
    def get_mass_error(self):
        '''
        Returns mass error used for feature matching (same as that stored in
        configuration settings)
        '''
        return float(self.config['MassError'])
    
    def set_adducts(self, adducts, header=0, sep=','):
        '''
        Set up adducts object to be used in feature matching. Pulls information
        needed from adducts.csv using the adduct names passed during
        configuration.
        
        Parameters
        ----------
        adducts: list of strings
            Adduct names to look up in adducts.csv
        header: int (optional)
            Line number to use as the header in adducts.csv. Zero-based index.
            All lines before will be ignored. Default = 0.
        sep: string (optional)
            Line delimiter for adducts.csv. Default = ','.

        Returns
        ----------
        adducts: DataFrame
            Final adducts to be used during feature matching.
        '''
        df = pd.read_csv('adducts.csv', header=header, sep=sep, low_memory=False)
        adducts = df[df['Name'].isin(adducts)]
        adducts.reset_index(inplace=True)
        self.adducts = adducts
        return self.adducts
    
    def get_adducts(self):
        '''
        Returns adducts object created during set_adducts().
        '''
        return self.adducts
    
    def get_blank_matched_features(self):
        '''
        Return empty matched_features DataFrame.
        '''
        return pd.DataFrame(columns=FEATURE_COLS)

#%% Data processing & downselection

    # Remove noise/contamination from MS data
    # TODO: break up this function.
    def downselect_features(self, df, indicator):
        '''
        Remove features from df that do not pass requirements for minimum
        number of samples, then creates a mean intensity column for the given
        sample.
        
        Parameters
        ----------
        df: DataFrame
            DataFrame containing features being considered for feature matching.
        indicator: string
            Substring used to indicate which columns belong to the sample of
            interest. Pulls multiple columns if more than one has this
            substring, and treats them as replicates.
        
        Returns
        ----------
        df: DataFrame
            Downselected DataFrame of features to use moving forward. Includes
            feature properties only.
        df_intensities: DataFrame
            Intensities for each feature across the repliactes, including a new
            column with the mean intensity.
        masses: list of floats
            Unique, sorted list of feature masses that passed downselection.
        '''

        df = df.copy()

        # Delete rows not passing min sample specs
        ind = df[df[FEATS_COL] < int(self.config['MinimumSamples'])].index
        df.drop(ind, inplace=True)
        
        # Create average intensity column
        df_intensities = self.get_cols(df, indicator, not_in='_ms2')
        df_intensities[MEAN_INT_COL] = df_intensities.mean(axis=1)
        
        if DEIMOS:
            usecols = [ID_COL, MZ_COL, CCS_COL, RT_COL]
            usecols.extend([x for x in df.columns if '_ms2' in x])
            df = df[usecols]
        else:
            df = df[[ID_COL, MZ_COL, CCS_COL]]
        
        # Unique list of masses (for searching)
        masses = list(set(df[MZ_COL]))
        masses.sort()
        
        return df, df_intensities, masses

    # Return cols to use for a given mix number.
    def get_cols(self, df, indicator, not_in=None):
        '''
        Return columns containing the given indicator (substring)
        '''
        if not_in is None:
            usecols = [x for x in list(df) if indicator in x]
        else:
            usecols = [x for x in list(df) if indicator in x and not_in not in x]
        return df[usecols]
    
    # Find values in cols with indicator above the given threshold
    # 1: above or equal to threshold, 0: below
    def get_cols_thresh(self, data, indicator, thresh, not_in=None):
        '''
        Count how many cells (in columns containing the indicator) are >= the
        given threshold.
        
        Parameters
        ----------
        df: DataFrame
            DataFrame containing columns to sum across
        indicator: string
            Substring used to indicate which columns to count across.
        thresh: float
            Value to use as a threshold.
        
        Returns
        ----------
        s: Pandas Series
            Series object containing the counts for each row.
        '''
        df = self.get_cols(data, indicator, not_in=not_in)
        for col in list(df):
            df[col] = np.where(df[col] >= thresh, 1, 0)
        return df.sum(axis=1)
    
    # Remove rows where blanks are higher than allowed number
    def remove_seen_in_blank(self, df):
        '''
        Remove features from df that do not pass requirements for maximum
        number of blanks.
        
        Parameters
        ----------
        df: DataFrame
            DataFrame containing features being considered for feature matching.
        
        Returns
        ----------
        df: DataFrame
            Downselected DataFrame of features to use moving forward. Includes
            all the same columns as originally passed in.
        '''
        ind = df[df[BLANKS_COL] > int(self.config['MaximumBlanks'])].index
        return df.drop(ind)
    
    # Convert feature MS2 string to list
    def parse_feature_ms2(self, s, sep=' '):
        ''' 
        Prepare feature MS2's for scoring. Built for DEIMoS output.
        '''
        if pd.isna(s):
            return None
        s = s.replace('[', '').replace(']', '')
        if len(s) > 0:
            s = [float(x) for x in s.split(sep)]
            return s
        return None

    # TODO: clean up this function/split into multiple
    def get_value(self, df, col_indic, i, allow_mult=True, return_df=True):
        '''
        Grab value(s) for given col(s) and row.
        
        Parameters
        ----------
        df: DataFrame
            DataFrame containing value(s) to be returned.
        col_indic: list of strings
            List of indicators (substrings) to search for in df column names.
        i: int
            Row to pull value(s) from.
        allow_mult: bool
            Whether multiple values are epxected to be returned. If False and
            more than one value found, prints error and returns None.
            Default = True.
        return_df: bool
            Whether a df is expected as a return type. If False, may return a
            Series object. Default = True.
        
        Returns
        ----------
        s or df: Pandas Series or DataFrame
            Values in the given columns/row. Type dependent on return_df
            variable. 
        '''
        
        # Narrow down column names
        col = df.columns
        for ci in col_indic:
            col = [x for x in col if ci in x]
    
        # Return if matched to a single column
        if allow_mult or len(col) == 1:
            if return_df:
                return df.loc[[i], col]
            return df.at[i, col[0]]  # Returns a value
        elif len(col) > 1:
            print('Too many columns matched')
        return None

#%% Feature matching!

    def _get_evidence_ccs(self, i, ccs_list, sample_info):
        '''
        For a given library entry and its CCS provided in the library, returns
        CCS errors for a given feature's CCS (i.e., compares the feature's CCS
        to all provided CCS from the library). Also returns the smallest of
        these errors in case RequireCCSMatch is set to True.
        '''
        
        # Collect errors
        ccs_errors = []
        for ccs in ccs_list:

            # Check the CCS value
            if ccs is not None:
                ccs_errors.append(mame_utils.percent_error(ccs,
                                  sample_info.loc[i, CCS_COL]))

        # Find smallest CCS error
        if np.sum(~pd.isna(ccs_errors)) > 0:
            min_ccs_error = np.nanmin(np.abs(ccs_errors))
        else:
            min_ccs_error = np.nan

        return ccs_errors, min_ccs_error

    def _get_evidence_ms2(self, i, ms2_df, sample_info, sample_int, min_int):
        '''
        For a given library entry and its MS2 provided in the library, returns
        MS2 sim scores for a given feature's MS2 (i.e., compares the feature's
        MS2 to all provided MS2 from the library). Cycles through all replicates
        (collision energies) and compares them to their respective collision
        energy from the library.
        
        Returns
        ----------
        scores: list of lists
            Calculated [10V scores, 20V scores, 40V scores] (e.g., if there
            are 4 10V MS2 available for this compound/adduct, 4 scores will be
            returned in the 10V scores list.) Does not return scores for
            collision energies this feature was not observed at (e.g., if the
            10V replicate has an intensity of 0, the 10V scores list will be
            empty.)
        '''
    
        # Build list of MS/MS similarities
        scores = [[], [], []]
        
        # Extract intensity column names
        usecols = list(sample_int)
        usecols.remove(MEAN_INT_COL)

        # Calculate set (10, 20, 40 V) of scores for each MS2 available
        for icol in usecols:

            # Check intensity for this sample
            if sample_int.loc[i, icol] >= min_int:

                # Select correct CE
                # TODO: Make agnostic to available CEs
                if self.config['MS2_10V_Indicator'] is not None and \
                self.config['MS2_10V_Indicator'] in icol:
                    ce_ind = 0
                    ce_name = '10V'
                elif self.config['MS2_20V_Indicator'] is not None and \
                self.config['MS2_20V_Indicator'] in icol:
                    ce_ind = 1
                    ce_name = '20V'
                elif self.config['MS2_40V_Indicator'] is not None and \
                self.config['MS2_40V_Indicator'] in icol:
                    ce_ind = 2
                    ce_name = '40V'
                else:
                    print('Error. Collision energy not found in file: %s' % icol)
                    return scores

                # Cycle through library MS2s
                # Assumes ints for each CE has already been averaged or only 1 replicate
                # Assumes new lines shown with \n
                # Could be multiple MS2's per type due to diff vendors
                l = []
                col = [x for x in ms2_df.columns if ce_name in x]
                if len(col) == 1:
                    l = ms2_df[col[0]].iloc[0]
                elif len(col) > 1:
                    print('Too many matched cols')
                    
                for ms2 in l:

                    # Check if there is an MS2 available
                    if type(ms2) == str and ms2 != 'nan':

                        # Parse into mass and intensity info
                        ms2_temp = ms2.replace('_x000D_', '')
                        ms2_temp = ms2_temp.replace('\r', '')
                        ms2_temp = ms2_temp.split(self.config['Lib_MS2_LineSplit'])
                        m1, i1 = mame_utils._parse_text_spectra(ms2_temp,
                                                                delimiter=self.config['Lib_MS2_ValueSplit'])

                        # Feature mass and intensities
                        # Force through same pipeline as lib ms2
                        # TODO: remove hack
                        name = icol.replace('_ms1', '')
                        feature_ms2_masses = self.get_value(sample_info,
                                                            [name, 'ms2_mz'],
                                                            i, allow_mult=False,
                                                            return_df=False)
                        feature_ms2_ints = self.get_value(sample_info,
                                                          [name, 'ms2_int'],
                                                          i, allow_mult=False,
                                                          return_df=False)
                        m2 = self.parse_feature_ms2(feature_ms2_masses)
                        i2 = self.parse_feature_ms2(feature_ms2_ints)
                        
                        if m2 is not None and i2 is not None:
                            s = mame_utils.convert_ms2_format(m2, i2)
                            m2, i2 = mame_utils._parse_text_spectra(s, minval=min_int)

                            # Calculate similarity
                            score = mame_utils.score_msms_similarity(m1, i1, m2, i2)

                            # Add this score to list
                            scores[ce_ind].append(score)
        return scores

    def get_evidence(self, mz, ccs_list, ms2_list, mass_list, sample_info,
                     sample_int, min_int, ms2_avail=True):
        '''
        Find all features that match to a given library entry, and return
        their associated errors.
        
        Parameters
        ----------
        mz: float
            Adduct mass of compound
        ccs_list: list of floats
            List of CCS provided in the library for this compound/adduct
        ms2_list: list of strings
            List of MS2 provided in the library for this compound/adduct
        mass_list: list of floats
            Sorted list of feature masses that passed the downselection step.
        sample_info: DataFrame
            Properties for all features being considered.
        sample_int: DataFrame
            Intensities for all features being considered, both as the
            intensities from individual replicates and a column for the mean
            intensity.
        ms2_avail: bool (optional)
            Whether MS2 is available in the given data. If False, MS2 scoring
            is not attempted. Default = True.
        
        Returns
        ----------
        new_rows: DataFrame
            Features matched to this compound/adduct, associated errors
            (where available: Mass Error, CCS Error, and MS2 Similarities),
            and feature information (Feature ID and Mean Intensity in the
            given sample)
        '''
        
        new_rows = self.get_blank_matched_features()
        
        # If adduct failed, exit here
        if mz is None:
            return new_rows
        
        mass_list = mass_list[:]  # List may be manipulated
        
        # Is this mass seen in the list of features?
        mz_in_list = mame_utils.in_list(mass_list, mz,
                                        e=float(self.config['MassError']) / 1000000.,
                                        sorted_list=True, a=False)
        
        # Add features with this mass to the list of compounds
        while mz_in_list is not None:
    
            # Get df index (actual attribute index)
            id_matches = sample_info.index[sample_info[MZ_COL] == mz_in_list]
            id_matches = id_matches.tolist()

            for i in id_matches:

                # Build list of CCS errors
                ccs_errors, min_ccs_error = self._get_evidence_ccs(i, ccs_list,
                                                                   sample_info)

                # Continue if mass alone is okay OR CCS was a match
                if not bool(self.config['RequireCCSMatch']) or \
                   min_ccs_error <= float(self.config['CCSError']):

                    if ms2_avail:
                        # Build lists of MS2 similarities
                        # TODO: further downselection by MS2 type (qtof, orbi, etc.)
                        scores = self._get_evidence_ms2(i, ms2_list, sample_info,
                                                        sample_int, min_int)
                    else:
                        scores = [0, 0, 0]

                    # Record feature match information
                    new_row = self.get_blank_matched_features()
                    new_row.at[0, 'Feature ID'] = sample_info.loc[i, ID_COL]
                    new_row.at[0, 'Mean Intensity'] = sample_int.loc[i, MEAN_INT_COL]
                    new_row.at[0, 'Mass Error'] = mame_utils.ppm_error(mz_in_list, mz)
                    new_row.at[0, 'CCS Error'] = ccs_errors
                    new_row.at[0, 'MS2 Similarity'] = scores

                    # Add this row to this compound's matched features
                    new_rows = pd.concat([new_rows, new_row])

            # Check for other close masses
            mass_list.remove(mz_in_list)  # Remove this feature mz from consideration
            mz_in_list = mame_utils.in_list(mass_list, mz,
                                            e=float(self.config['MassError']) / 1000000.,
                                            sorted_list=True, a=False)

        return new_rows

    # For all library entries, find all features from the experimental data 
    # that match
    # TODO: Separate pos and neg adducts/data (include pos or neg as input)
    # TODO: Make agnostic to mass/ccs/msms/etc. available (?)
    def feature_matching(self, data, sample_names, mode, ms2_avail=True):
        '''
       For all library entries, find all features from the experimental data 
       that match to each.
        
        Parameters
        ----------
        data: DataFrame
            Experimental data. DataFrame should be generated from the
            import_data() function.
        mix_names: list of strings
            List of samples names to perform feature matching for.
        mode: string
            Must be 'Pos' or 'Neg', representing this data comes from positive
            or negative mode, respectively.
        ms2_avail: bool (optional)
            Whether MS2 is available in the given data. If False, MS2 scoring
            is not attempted. Default = True.
        
        Returns
        ----------
        matched_features: DataFrame
            Features matched to compounds in the library, for the samples of
            interest.
        '''
        data = data.copy()
        matched_features = self.get_blank_matched_features()
        
        if mode.lower() == 'pos':
            min_int = float(self.config['PosModeMinInt'])
            adducts = self.adducts[self.adducts['Charge'] > 0]
        elif mode.lower() == 'neg':
            min_int = float(self.config['NegModeMinInt'])
            adducts = self.adducts[self.adducts['Charge'] < 0]
        else:
            print('Invalid mode')
            return matched_features
        
        # Remove features seen in blanks above min allowable times
        if self.config['BlankIndicator'] is not None:

            data[BLANKS_COL] = self.get_cols_thresh(data,
                                                    self.config['BlankIndicator'],
                                                    min_int)
            data = self.remove_seen_in_blank(data)

        # Cycle through sample features
        for sample in sample_names:

            # Columns associated with this mix
            data[FEATS_COL] = self.get_cols_thresh(data, sample, min_int, 
                                                   not_in='_ms2')
            
            # Get & process data for this mix
            feature_info, feature_int, feature_masses = self.downselect_features(data,
                                                                                 sample)

            # Begin cycling through each library entry and its possible matches
            for cpd_no in range(len(self.lib_formulas)):

                # Initalize compound information
                formula = self.lib_formulas[cpd_no]

                # TODO: change to always calling adduct by name, no assumed index
                # If prop not available for that adduct, make it None
                for j, adduct in adducts.iterrows():
        
                    # Find features that match to this sample + cpd + adduct
                    mz = mame_utils.calc_new_mz(formula, adduct)
                    try:
                        ccs = self.lib_ccs.at[cpd_no, '%s CCS' % adduct['Name']]
                    except KeyError:
                        ccs = [np.nan]
#                    ms2_cols = [x for x in self.lib_msms.columns if adduct['Name'] in x]
#                    ms2 = self.lib_msms[ms2_cols].iloc[[cpd_no]]  # Grab all CE's
                    ms2 = self.get_value(self.lib_msms, [adduct['Name']], cpd_no)
                    new_rows = self.get_evidence(mz, ccs, ms2, feature_masses[:],
                                                 feature_info, feature_int,
                                                 min_int, ms2_avail=ms2_avail)

                    # Label this evidence as being from this compound + adduct
                    try:
                        if len(new_rows) > 0:
                            new_rows['Sample'] = sample
                            new_rows['Cpd No'] = cpd_no
                            new_rows['Adduct'] = adduct['Name']
                            
                            # Append to all feature evidence
                            matched_features = pd.concat([matched_features,
                                                          new_rows])
                    except TypeError:
                        pass

            del data[FEATS_COL]
            
        return matched_features.reset_index(drop=True)

#%% Import experimental data

# Pull data from DEIMoS or MassProfiler input
# MS2 = True if MS2 available in data
def import_data(fname, intensity_indic, ms2=True, rt=True):
    '''
    Import experimental data from feature file. If global variable DEIMOS
    is set to True, assumes a certain format. If DEIMOS = False, assumes
    the format is the same as what comes from MassProfiler (note, MassProfiler
    compatibility is depricated and no longer guarenteed to work as expected).
    
    Parameters
    ----------
    fname: string
        Path to data file.
    intensity_indic: string
        Substring that indicates a given column represents MS1 intensities.
    ms2: bool (optional)
        Whether MS2 data is available in this data file. If False, does not
        attempt to pull MS2 data. Default = True.
    rt: bool (optional)
        Whether retention time data is available in this data file. If False,
        does not attempt to pull retention time data. Default = True.
    
    Returns
    ----------
    df: DataFrame
        Feature data.
    '''

    # Default column names from given tools
    if DEIMOS:
        header = 0
        sep = '\t'
        id_col = 'feature_idx'
        mass_col = 'mz'
        ccs_col = 'ccs'
        rt_col = 'retention_time'
        ms2_col1 = 'ms2_mz'
        ms2_col2 = 'ms2_intensity'

        usecols = [id_col, mass_col, ccs_col]
        
        if rt:
            usecols.extend([rt_col])
        if ms2:
            usecols.extend([ms2_col1, ms2_col2])

    else:  # Assume MassProfiler & no MS2 or RT available
        header = 4
        sep = ','
        id_col = 'ID'
        mass_col = 'm/z'
        ccs_col = 'CCS'
        usecols = [id_col, mass_col, ccs_col]

    # Read in data
    df = pd.read_csv(fname, header=header, sep=sep, low_memory=False)
    
    # Rename index column if needed
    if 'feature_idx' not in df.columns:
        if 'Unnamed: 0' in df.columns[0]:
            df.rename(columns={'Unnamed: 0':'feature_idx'}, inplace=True)
        elif 'feature' in df.columns[0]:
            df.rename(columns={'feature':'feature_idx'}, inplace=True)
        elif 'id' in df.columns[0]:
            df.rename(columns={'id':'feature_idx'}, inplace=True)


    # Select necessary columns
    int_cols = [x for x in list(df) if intensity_indic in x]
    if 'ms2_intensity' in int_cols:
        int_cols.remove('ms2_intensity')
    usecols.extend(int_cols)
    df = df[usecols]
    
    # Zero out ~0 intensity values
    if not DEIMOS: # MP uses 0.001 as its 0 value
        for col in int_cols:
            df[col] = np.where(df[col] == 0.001, 0, col)
    
    # Rename to keep consistent downstream
    df = df.rename(columns={id_col: ID_COL, mass_col: MZ_COL,
                            ccs_col: CCS_COL})
    if DEIMOS and rt:
        df = df.rename(columns={rt_col: RT_COL})
        
    return df
