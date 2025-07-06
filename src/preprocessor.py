'''
Preprocessor script.

The script performs the preprocessing of a text column within a pandas DataFrame.
The input column contains a label representing either the real name of a company,
or a custom label referring to a specific company, not necessarely identical or
similar to the real name.
The output is a DataFrame containing two new columns, representing a chain of
components (shingles) of the original input column. The difference between
the two columns is the application of an additional step of filtering to
remove the most frequent words.

The class requires "pandas" as external package.
'''

import logging
import pandas as pd
from tools import preprocess_label
from tokenizer import Tokenizer
from shingler import Shingler
from environment_manager import LocalEnvironmentManager, RemoteEnvironmentManager


logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()


class Preprocessor():
    '''
    The class encapsulating all the tasks required for preprocessing
    a specific text column in a dataframe.
    '''

    __slots__ = (
        "_input_df",
        "_entity_label",
        "_environment_manager",
        "_discarded_token_path",
        "_storage_options",
        "_grouping_key",
        "_tokenizer",
        "_shingler")

    def __init__(self,
                 input_df: pd.DataFrame,
                 entity_label: str,
                 environment_manager: LocalEnvironmentManager | RemoteEnvironmentManager):
        self._input_df = input_df
        self._entity_label = entity_label
        self._environment_manager = environment_manager
        self._discarded_token_path = self._environment_manager.discarded_token_path
        self._storage_options = self._discarded_token_path.storage_options
        self._grouping_key = None
        self._tokenizer = None
        self._shingler = None

    @property
    def input_df(self) -> pd.DataFrame:
        '''
        The dataframe containing the entity to preprocess.
        '''
        return self._input_df

    @property
    def entity_label(self) -> str:
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

    @property
    def environment_manager(self):
        '''
        The environment manager to be used for dealing
        with the I/O steps (either local or remote).
        '''
        return self._environment_manager

    @property
    def discarded_token_path(self):
        '''
        The path (defined as universal path) to the
        position of the discarded token parquet file.
        '''
        return self._discarded_token_path

    @property
    def storage_options(self):
        '''
        The dictionary containing the storage options
        for storing the discaded_token parquet file.
        '''
        return self._storage_options

    def _preprocess_entity_labels(self):
        '''
        Preprocess the column entity_label.
        '''
        # Defining the name of the output column
        output_column_name = f'preprocessed_{self._entity_label}'
        # Applying the preprocessing steps to each element of entity_label
        self._input_df[output_column_name] = [preprocess_label(x) for x in self._input_df[
            self._entity_label]]

    def _run_tokenizer(self,
                       legal_entities: list[str],
                       languages: list[str]=None):
        '''
        Initialize the dataset to tokenize and perform
        the whitespace-based tokenization.
        '''
        if languages is None:
            languages = ['french', 'english']
        self._preprocess_entity_labels()
        # Splitting denomination by token
        logger.info('Tokenizing the %s', self._entity_label)
        # Initialising the dataset for the tokenizer
        self._tokenizer.set_dataset_for_tokenizer(self._input_df)
        # Running all the steps for the tokenizer
        self._tokenizer.run(legal_entities, languages)
        # Recovering the dataframe with the tokenizer label
        df_token = self._tokenizer.df_entity_token
        return df_token

    def _run_shingler(self,
                      df_token: pd.DataFrame,
                      discarded_token: pd.DataFrame):
        '''
        Initialize the dataset to shingle and perform
        the shingling task.
        '''
        # Initialising the dataset for the shingler
        self._shingler.set_dataset_for_shingling(df_token)
        # Shingling the tokens
        self._shingler.shingle()
        logger.info('Defining the shingled %s', self._entity_label)
        df_output = self._shingler.create_shingled_entities(discarded_token['token'])
        if len(df_output) == 0:
            return df_output
        # If the filtered shingled entitiy is completely empty, fill it with the unfiltered one
        df_output.fillna({f'shingled_{self._entity_label}_filtered':
                          df_output[f'shingled_{self._entity_label}_unfiltered']},
                         inplace=True)
        return df_output


class TrainingPreprocessor(Preprocessor):
    '''
    The specialized preprocessor class for the training phase.
    '''
    def __init__(self,
                 input_df,
                 entity_label,
                 environment_manager):
        super().__init__(input_df,
                         entity_label,
                         environment_manager)
        self._grouping_key = ['siret', 'denomination_type']
        self._tokenizer = Tokenizer(self._entity_label,
                                    self._grouping_key)
        self._shingler = Shingler(self._entity_label,
                                  self._grouping_key)

    def _set_lsh_index(self, shingled_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Assign a unique index to each shingled entity.
        If the shingled entity is empty assign -1.
        
        :param shingled_df: pandas DataFrame. The dataframe containing
        a column with the shingled entities.
        
        return shingled_df: pandas DataFrame. The input dataframe with an
        additional column containing the unique index.
        '''
        input_column_name = f'shingled_{self._entity_label}_filtered'
        shingled_df["lsh_index"] = shingled_df.groupby(input_column_name).ngroup()
        shingled_df.fillna({input_column_name: '',
                            'lsh_index': -1},
                            inplace=True)
        shingled_df['lsh_index'] = shingled_df['lsh_index'].astype(int)
        return shingled_df

    def _prepare_personne_morale(self, denomination_entities: list[str]):
        '''
        Performing preprocessing steps for the moral entities denominations

        :param denomination_entities: list. The name of the columns in
        the dataframe containing denominations related to moral entities
        '''
        # Unpivoting the dataframe
        self._input_df = self._input_df.melt(id_vars=['siren', 'siret'],
                                             value_vars=denomination_entities,
                                             var_name='denomination_type',
                                             value_name='denomination')
        # Dropping all the duplicated denomination for the same siren
        self._input_df.drop_duplicates(subset=['siren', 'denomination'], inplace=True)
        # Dropping None
        self._input_df.dropna(subset='denomination',
                              inplace=True,
                              ignore_index=True)
        # Keeping only the SIRET as identifier of a company/etablissement
        self._input_df.drop(columns='siren', inplace=True)

    def run(self,
            legal_entities: list[str],
            denomination_entities: list[str],
            languages: list[str]=None):
        '''
        Perform all the steps for the preprocessing of the entity label.
        
        :param legal_entitites: list. The list of string representing
        legal entities.
        :param languages: list. The list of languages to be used for
        defining the stopwords.

        return df_ouput: pandas DataFrame. The dataframe containing
        the preprocessed entity.
        '''
        logger.info('Preparing the dataset of the moral entities')
        self._prepare_personne_morale(denomination_entities)
        df_token = self._run_tokenizer(legal_entities, languages)
        # Creating or loading the dataset with the most frequent tokens
        logger.info('Extracting token to filter')
        discarded_token = self._tokenizer.get_token_to_filter()
        discarded_token.to_parquet(self._discarded_token_path,
                                    engine='pyarrow',
                                    compression='gzip',
                                    storage_options=self._storage_options)
        logger.info('Shingling the tokens')
        df_output = self._run_shingler(df_token,
                                       discarded_token)
        logger.info('Setting the unique indexes for the shingled, filtered %s',
                    self._entity_label)
        df_output = self._set_lsh_index(df_output)
        return df_output

class PredictionPreprocessor(Preprocessor):
    '''
    The specialized preprocessor class for the prediction phase.
    '''
    def __init__(self,
                 input_df,
                 entity_label,
                 environment_manager):
        super().__init__(input_df,
                         entity_label,
                         environment_manager)
        self._grouping_key = ['COMPAUXLIB']
        self._tokenizer = Tokenizer(self._entity_label,
                                    self._grouping_key)
        self._shingler = Shingler(self._entity_label,
                                  self._grouping_key)

    def run(self,
            legal_entities: list[str],
            languages: list[str]=None):
        '''
        Perform all the steps for the preprocessing of the entity label.
        
        :param legal_entitites: list. The list of string representing
        legal entities.
        :param languages: list. The list of languages to be used for
        defining the stopwords.

        return df_ouput: pandas DataFrame. The dataframe containing
        the preprocessed entity.
        '''
        logger.info('Preprocessing entities')
        self._preprocess_entity_labels()
        logger.info('Tokenizing entities')
        df_token = self._run_tokenizer(legal_entities, languages)
        if len(df_token) == 0:
            logger.warning('Tokenization returns an empty list.')
            return df_token
        # Creating or loading the dataset with the most frequent tokens
        logger.info('Loading the tokens to filter')
        discarded_token = pd.read_parquet(self._discarded_token_path,
                                          engine='pyarrow',
                                          storage_options=self._storage_options)
        logger.info('Shingling the tokens')
        df_output = self._run_shingler(df_token,
                                       discarded_token)
        return df_output
