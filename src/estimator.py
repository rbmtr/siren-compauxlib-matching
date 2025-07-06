'''
Estimator of the final results.

The script evaluates the jaccard similarity between COMPAUXLIB and
denominationUniteLegale for the SIRENs obtained by querying the LSH.
It then defines some entries as "valid" if the jaccard similarity 
between the labels keeping all the token but the stopwords is
equal to 1 (called the "unfiltered" case, in contrast with the case
where the most frequent tokens are removed, called "filtered"). Are
also considered "valid" all the first N entries above a fixed threshold.

The classes require "pandas" and "nltk" as external package.
'''
import logging
import pandas as pd
from numpy.linalg import norm

logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()

INFINITE_DISTANCE = 1e8
    
class Estimator():
    '''
    The base class collecting all the methods necessary for
    evaluating the candidates to propose.
    '''

    __slots__ = (
        "_input_df",
        "_personne_morale_df",
        "_fec_coordinates",
        "_denomination_unite_legale",
        "_denomination_etablissement",
        "_threshold",
        "_nb_rank"
        )

    def __init__(self,
                 input_df: pd.DataFrame,
                 personne_morale_df: pd.DataFrame,
                 denomination_unite_legale: list[str],
                 denomination_etablissement: list[str],
                 threshold: int,
                 nb_rank: int,
                 fec_coordinates: pd.DataFrame=None
                 ):
        self.input_df = input_df
        self._personne_morale_df = personne_morale_df
        self._fec_coordinates = fec_coordinates
        self._denomination_unite_legale = denomination_unite_legale
        self._denomination_etablissement = denomination_etablissement
        self._threshold = threshold
        self._nb_rank = nb_rank

    @property
    def input_df(self):
        '''
        The dataframe from which extracting the
        SIRENs to propose for each COMPAUXLIB.
        '''
        return self._input_df
    @input_df.setter
    def input_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('input_df must be a pandas DataFrame')
        for case in ['filtered', 'unfiltered']:
            denomination_entity = f'shingled_denomination_{case}'
            compauxlib_entity = f'shingled_COMPAUXLIB_{case}'
            if denomination_entity not in value.columns:
                raise ValueError(f'input_df must contains the column {denomination_entity}')
            if compauxlib_entity not in value.columns:
                raise ValueError(f'input_df must contains the column {compauxlib_entity}')
        self._input_df = value

    def _prepare_final_output(self,
                               df_to_prepare: pd.DataFrame,
                               columns_to_drop: list,
                               columns_to_deduplicate: list,
                               joining_key: str):
        '''
        Clean up the input dataframe and merge it with the informations from
        the personne morale base
        '''
        df_to_prepare.drop(columns=columns_to_drop, inplace=True)
        df_to_prepare.drop_duplicates(subset=columns_to_deduplicate, inplace=True)
        df_to_prepare = df_to_prepare.merge(self._personne_morale_df,
                                            how='left',
                                            on=joining_key)
        return df_to_prepare

    def _prepare_unite_legale(self):
        '''
        Extract the cases where the highest denomination is one referring to the unite legale.
        Evaluates the sirens and join all the informations from the personne
        morale base to each siren.
        '''
        df_fec_unite_legale = self.input_df[self.input_df['denomination_type'
                                                          ].isin(self._denomination_unite_legale)
                                            ].reset_index(drop=True)
        df_fec_unite_legale['siren'] = df_fec_unite_legale['siret'].apply(lambda x: x[:9])
        siret_to_discard = df_fec_unite_legale['siret'].unique()
        df_fec_unite_legale = self._prepare_final_output(df_fec_unite_legale,
                                                         ['siret', 'denomination_type'],
                                                         ['COMPAUXLIB', 'siren'],
                                                         'siren')
        return df_fec_unite_legale, siret_to_discard

    def _prepare_etablissement(self, siret_to_discard: list):
        '''
        Extract the cases where the highest denomination is one referring to the etablissement.
        Join all the informations from the personne morale base to each siret.
        '''
        df_fec_etablissement = self.input_df[~self.input_df['siret'].isin(siret_to_discard)
                                             ].reset_index(drop=True)
        df_fec_etablissement = self._prepare_final_output(df_fec_etablissement,
                                                          ['denomination_type'],
                                                          ['COMPAUXLIB', 'siret'],
                                                          'siret')
        return df_fec_etablissement

    def _set_geo_distance(self):
        '''
        Evaluate the distance between each candidate site and the FEC site.
        '''
        valid_input_df = self.input_df[self.input_df['candidate_lsh_index'] != -1]
        valid_input_df['geo_distance'] = INFINITE_DISTANCE
        for site in range(len(self._fec_coordinates)):
            valid_input_df['shifted_abscisse'] = valid_input_df['AbscisseEtablissement'
                                                                 ]-self._fec_coordinates[
                                                                     'Abscisse'].values[site]
            valid_input_df['shifted_ordonnee'] = valid_input_df['OrdonneeEtablissement'
                                                                ]-self._fec_coordinates[
                                                                    'Ordonnee'].values[site]
            valid_input_df['geo_distance'] = valid_input_df.apply(lambda x: min(
                norm([x['shifted_abscisse'],
                      x['shifted_ordonnee']]),
                      x['geo_distance']),
                      axis=1)
        self.input_df = self.input_df.merge(valid_input_df[['geo_distance']],
                                            left_index=True,
                                            right_index=True)

    def _set_final_rank(self, limit):
        '''
        It then rank the candidates for each COMPAUXLIB by the unfiltered
        Jaccard distance and the geographical distance (the closest the
        higher the rank).
        Finally it set a label validated to the first 'limit' entries
        (per COMPAUXLIB).
        '''
        self.input_df['final_rank'] = 1
        # Defining a dummy column geo_distance
        self.input_df['geo_distance'] = 0
        if self._fec_coordinates is not None:
            # If fec coordinates are provided, drop the geo_distance
            # column since it is evaluated in set_geo_distance
            self.input_df.drop(columns=['geo_distance'], inplace=True)
            self._set_geo_distance()
        self.input_df.sort_values(['COMPAUXLIB',
                                   'jaccard_distance_unfiltered',
                                   'geo_distance'],
                                  inplace=True)
        self.input_df['final_rank'] = self.input_df.groupby('COMPAUXLIB')['final_rank'].cumsum()
        self.input_df.sort_values(by='final_rank', inplace=True)
        self.input_df.drop_duplicates(subset='siren', inplace=True)
        self.input_df.reset_index(drop=True, inplace=True)
        self.input_df['validated'] = True
        if limit is not None:
            self.input_df['validated'] = self.input_df['final_rank'] <= limit
            self.input_df.reset_index(drop=True, inplace=True)
            
    def run(self, limit=None):
        '''
        Perform all the steps for extracting the list of SIRENs to be associated
        to the COMPAUXLIB.
        '''
        # Pretreat the entries matching to denominations belonging to unite legale
        df_fec_unite_legale, siret_to_discard = self._prepare_unite_legale()
        # Pretreat the entries matching to denominations belonging to etablissements
        df_fec_etablissement = self._prepare_etablissement(siret_to_discard)
        # Merging the two results
        self.input_df = pd.concat([df_fec_unite_legale,
                                   df_fec_etablissement])
        # Concatenating the list of shingles
        columns_to_preprocess = ['shingled_COMPAUXLIB_unfiltered',
                                 'shingled_COMPAUXLIB_filtered',
                                 'shingled_denomination_unfiltered',
                                 'shingled_denomination_filtered']
        self.input_df[columns_to_preprocess] = self.input_df[columns_to_preprocess].map(
            lambda x: ','.join(sorted(x)), na_action='ignore')
        self.input_df.fillna({column: '' for column in columns_to_preprocess}, inplace=True)
        self.input_df.drop_duplicates(inplace=True, ignore_index=True)
        self._set_final_rank(limit)
