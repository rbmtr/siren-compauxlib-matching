'''
Tokenizer script.

The script contains the Stopwords and Tokenizer classes.

The Stopwords class prepares the list of stopwords to be used within the tokenizer.
It provides additional methods for recovering stopwords with specifics characteristics.

The Tokenizer class performs the tokenization and preprocessing of a text column
within a pandas DataFrame.
The input column contains a label representing either the real name of a company,
or a custom label referring to a specific company, not necessarely identical or
similar to the real name.
The input dataframe is modified in place, ending up with a token column, containing
the preprocessed tokens composing the original input column.

The class requires "pandas" and "nltk" as external package.
'''

import logging
import sys
import pandas as pd
from nltk.corpus import stopwords
from tools import preprocess_label, tokenize, get_label_length, get_labels_mask

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()


class Stopwords():
    '''
    The class initialize the list of stopwords and
    contains methods to operate on it.
    '''

    __slots__ = ("_stopwords",)

    def __init__(self):
        self._stopwords = []

    @property
    def stopwords(self) ->list[str]:
        '''
        The list of stopwords.
        '''
        return self._stopwords

    @stopwords.setter
    def stopwords(self, value):
        if not isinstance(value, list):
            raise TypeError('stopwords must be a list')
        if not all(isinstance(element, str) for element in value):
            raise TypeError('The list contains elements that are not string')
        self._stopwords = value

    def instantiate_stopwords(self,
                              legal_entities: list[str],
                              languages: list[str]):
        '''
        Initialize the list of stopwords based on the list
        of legal entities and of languages to be used.
        
        :param legal_entities: list. The list of legal entities.
        :param languages: list. The list of languages to be used for defining the stopwords.        
        '''
        languages_stopwords = []
        for language in languages:
            languages_stopwords.extend(stopwords.words(language))
        self._stopwords = set(preprocess_label(
            word) for word in languages_stopwords + legal_entities)

    def get_multi_character_stopwords(self):
        '''
        Extract the stopwords composed by multiple characters.
        
        return list. The list of multi character's stopwords.
        '''
        return [word for word in self._stopwords if len(word) > 1]


class Tokenizer():
    '''
    The class encapsulate the different methods for tokenizing and
    preprocessing the tokens.
    '''

    __slots__ = (
        "df_entity_token",
        "stopwords",
        "_entity_label",
        "_grouping_key",
        "_preprocessed_entity_label"
        )
    def __init__(self,
                 entity_label: str,
                 grouping_key: list):
        self.df_entity_token = None
        self.stopwords = None
        self._entity_label = entity_label
        self._grouping_key = grouping_key
        self._preprocessed_entity_label = f'preprocessed_{self._entity_label}'

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

    def _get_tokens_frequencies(self):
        '''
        Evaluates the frequency of each token in the dataframe.

        return pandas DataFrame. A DataFrame containing a column
        with the tokens and a column with the associated frequency.
        '''
        df_token_ranking = self.df_entity_token.groupby(by='token',
                                                        as_index=False).count(
                                                            ).rename(columns={'siret': 'frequency'})
        df_token_ranking['frequency'] = df_token_ranking['frequency'
                                                         ]/df_token_ranking['frequency'].sum()
        return df_token_ranking

    def get_token_to_filter(self,
                            method: str='quantile',
                            threshold: float | int=0.9999) -> pd.DataFrame:
        '''
        Filtering the input dataset excluding the most recurring token,
        defined according to the method and threshold
        parameters. Dumping the list of discarded token.

        :param method: str. The type of method to use for filtering
        the most frequent tokens.
        :param threshold: float, int. The value of the threshold for
        discarding the tokens.

        return pandas DataFrame. A DataFrame containing a column with the
        discarded token and a column with the associated frequency.
        '''
        df_ranking = self._get_tokens_frequencies()
        df_ranking = df_ranking[['token', 'frequency']]
        if method == 'quantile':
            if (threshold > 1) or (threshold < 0):
                logger.error('Method quantile requires a threshold between 0 and 1.')
                sys.exit(1)
            quantile_threshold = df_ranking['frequency'].quantile(threshold)
            discarded_token = df_ranking[df_ranking['frequency'] >= quantile_threshold]
        elif method == 'topk':
            if threshold < 1:
                logger.error('Method topk requires a threshold greater or equal to 1.')
                sys.exit(1)
            df_ranking.sort_values('frequency',
                                   ascending=False,
                                   inplace=True)
            discarded_token = df_ranking.head(threshold)
        else:
            logger.error('''
                         Invalid method chosen for filtering.
                         Admitted methods are: quantile, topk
                         ''')
            sys.exit(1)
        return discarded_token

    def set_dataset_for_tokenizer(self,
                                  input_df: pd.DataFrame):
        '''
        Initialize the dataframe on which apply the tokenizer.

        :param input_df: pandas DataFrame. The DataFrame from
        which recover the necessary sub-DataFrame for the tokenizer.
        '''
        # Extract relevant dataframe
        self.df_entity_token = input_df[self._grouping_key + [self._preprocessed_entity_label]]

    def _set_stopwords(self,
                       legal_entities: list[str],
                       languages: list[str]):
        '''
        Initialize the stopwords for the tokenizer.

        :param legal_entities: list. The list with the
        legal entities of the companies.
        :param languages: list. The list with the languages to be used
        for defining the stopwords.
        '''
        self.stopwords = Stopwords()
        self.stopwords.instantiate_stopwords(legal_entities, languages)

    def _tokenize(self):
        '''
        Split the content of a DataFrame column by
        the whitespace and expand the resulting list in a DataFrame column.
        '''
        # Tokenizeing the entity label
        self.df_entity_token['token'] = [tokenize(
            token) for token in self.df_entity_token[self._preprocessed_entity_label]]
        self.df_entity_token = self.df_entity_token.explode('token', ignore_index=True)
        self.df_entity_token.drop(columns=[self._preprocessed_entity_label], inplace=True)
        self.df_entity_token.dropna(subset='token', inplace=True)

    def _set_token_length(self):
        '''
        Defines a new column in a DataFrame containing the
        length of the token.
        '''
        # Extracting the number of characters from each token
        self.df_entity_token['token_length'] = [get_label_length(
            token) for token in self.df_entity_token['token']]

    def _set_max_token_length(self):
        max_token_length = self.df_entity_token.groupby(self._grouping_key,
                                                        as_index=False)['token_length'].max()
        max_token_length.rename(columns={'token_length': 'max_token_length'}, inplace=True)
        self.df_entity_token = self.df_entity_token.merge(max_token_length,
                                                          how='left',
                                                          on=self._grouping_key)

    def _aggregate_single_characters_entities(self):
        single_characters_df = self.df_entity_token[
            self.df_entity_token['max_token_length'] == 1].reset_index(drop=True)
        single_characters_df = single_characters_df.groupby(self._grouping_key,
                                                            as_index=False)[
                                                                'token'].apply(''.join)
        return single_characters_df

    def _remove_multi_characters_stopwords(self):
        '''
        Remove from the attribute DataFrame the rows containing tokens belonging to
        the list of stopwords composed by multiple characters.
        '''
        multi_characters_stopwords = self.stopwords.get_multi_character_stopwords()
        multi_characters_stopwords_mask = get_labels_mask(self.df_entity_token['token'].tolist(),
                                                          multi_characters_stopwords)
        multi_characters_stopwords_mask = [not mask for mask in multi_characters_stopwords_mask]
        # Removing multi characters stopwords from token
        self.df_entity_token = self.df_entity_token[
            multi_characters_stopwords_mask].reset_index(drop=True)

    def _preprocess_single_characters_entities(self):
        '''
        Merge together the tokens of the entities composed by only single
        character tokens.
        '''
        self._set_token_length()
        self._set_max_token_length()
        single_characters_df = self._aggregate_single_characters_entities()
        multi_characters_df = self.df_entity_token[
            self.df_entity_token['token_length'] > 1].reset_index(drop=True)
        self.df_entity_token = pd.concat([single_characters_df,
                                          multi_characters_df]
                                         ).reset_index(drop=True)
        self.df_entity_token.drop(columns=['token_length', 'max_token_length'], inplace=True)

    def _preprocess_tokenized_dataset(self):
        '''
        Apply the different steps of preprocessing on the tokens.
        '''
        self._remove_multi_characters_stopwords()
        self._preprocess_single_characters_entities()

    def run(self, legal_entities, languages):
        '''
        Tokenize and preprocess the tokens.

        :param legal_entities: list. The list of legal entities
        of the companies.
        :param languages: list. The list of the languages to be used
        for defining the stopwords.
        '''
        self._tokenize()
        self._set_stopwords(legal_entities, languages)
        self._preprocess_tokenized_dataset()
