'''
Shingler script.

The script performs the shingling of a text column within a pandas DataFrame.
Shingling means applying a rolling window of fixed size on the text and
extracting the characters contained in the window.
The shingled token are then recomposed in a single string, separated by ','
in order to be suitable for storing in a database.
The input column contains a token (a single word without necessarely a meaning).
The output is a DataFrame containing two new columns, representing a chain of
components (shingles) of the original input column. The difference between
the two columns is the application of an additional step of filtering to
remove the most frequent words.

The class requires "pandas" as external package.
'''

import logging
import pandas as pd
from tools import shingle_label, chain_shingles, get_labels_mask

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()

class Shingler():
    '''
    The class defining a shingle and all the operations related to it.
    '''

    __slots__ = (
        "_entity_label",
        "_window",
        "_grouping_key",
        "df_shingled"
        )

    def __init__(self, entity_label: str, grouping_key: list, window: int=4):
        self._window = window
        self._entity_label = entity_label
        self._grouping_key = grouping_key
        self.df_shingled = None

    @property
    def window(self) -> int:
        '''
        The maximum size of the rolling window
        for obtaining the shingles.
        '''
        return self._window

    @property
    def entity_label(self):
        '''
        The text column to preprocess.
        '''
        return self._entity_label

    @property
    def grouping_key(self):
        '''
        The text column to be used to aggregate
        the preprocessed entity.
        '''
        return self._grouping_key

    def set_dataset_for_shingling(self, input_df: pd.DataFrame):
        '''
        Extract a sub-dataframe from the input one.

        :param input_df: pandas DataFrame. The
        dataframe containing the tokens to shingle.
        '''
        self.df_shingled = input_df[self._grouping_key + ['token']]

    def shingle(self):
        '''
        Define a new column in the attribute dataframe
        containing the set of shingled tokens.
        '''
        self.df_shingled['shingled_token'] = [shingle_label(token, self._window)
                                              for token in self.df_shingled['token']]

    def _create_shingled_entity(self, df_token: pd.DataFrame):
        '''
        Set in a new column the chain of shingles for each entity
        defined by the grouping key.

        :param df_token: pandas DataFrame. The DataFrame containing the shingled tokens.

        return pandas DataFrame. The DataFrame containing the chain
        of shingle for each grouping key.
        '''
        df_preprocessed = df_token.groupby(by=self._grouping_key,
                                           as_index=False)[
                                               'shingled_token'
                                               ].apply(chain_shingles)
        df_preprocessed.rename(columns={'shingled_token': f'shingled_{self._entity_label}'},
                               inplace=True)
        return df_preprocessed

    def create_shingled_entities(self, discarded_token: pd.Series) -> pd.DataFrame:
        '''
        The full pipeline taking a DataFrame containing the tokens and creating two
        chains of shingles for each entity, one with the full ensemble of shingled tokens,
        the other discarding the list of token passed as input.

        :param discarded_token: pandas DataFrame. The DataFrame containing
        the tokens to discard and the associated frequency.

        return pandas DataFrame. The DataFrame containing for each entity
        the two chained shingled tokens, filtered and non-filtered.
        '''
        df_preprocessed = self._create_shingled_entity(self.df_shingled)
        logger.info('Defining the shingled %s with the filtered tokens', self._entity_label)
        discarded_mask = get_labels_mask(self.df_shingled['token'].tolist(), discarded_token.tolist())
        discarded_mask = [not mask for mask in discarded_mask]
        df_filtered = self.df_shingled[discarded_mask].reset_index(drop=True)
        if len(df_filtered) == 0:
            logger.warning("The COMPAUXLIB to query contains only filtered tokens.")
            return df_filtered
        df_preprocessed_filtered = self._create_shingled_entity(df_filtered)
        df_preprocessed = df_preprocessed.merge(df_preprocessed_filtered,
                                                how='left',
                                                on=self._grouping_key,
                                                suffixes=('_unfiltered','_filtered'))
        return df_preprocessed
