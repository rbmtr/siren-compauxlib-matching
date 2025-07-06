'''
LSH preprocessor and execution script.

The script performs the preprocessing of a pandas DataFrame in order to prepare it
for the creation or exploitation of an index for the Locality Sensitive Hashing.

The input column for creating the index contains a chain of text components
(shingles) extracted from the real name of a company or a custom label referring
to a specific company, not necessarely identical or similar to the real name.

In the training part, the input column is used for creating the LSH, in the
prediction part the input column is used for querying similar labels from the
LSH.

The classes require "pandas" and "datasketch" as external package.
'''
import logging
import json
import pandas as pd
from fsspec import filesystem
from xxhash import xxh32_intdigest
from datasketch.minhash import MinHash
from datasketch.lean_minhash import LeanMinHash
from custom_lsh import CustomLSH
from environment_manager import LocalEnvironmentManager, RemoteEnvironmentManager

logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()


class BaseLSHProcessor():
    '''
    The base class encapsulating the attributes and methods required for
    constructing and exploiting the LSH.
    '''

    __slots__ = (
        "_input_df",
        "_entity_label",
        "_num_permutations",
        "_threshold",
        "_minhashes",
        "_lsh",
        "_env_manager"
        )

    def __init__(self,
                 input_df: pd.DataFrame,
                 entity_label: str,
                 num_permutations: int,
                 threshold: int,
                 env_manager: LocalEnvironmentManager | RemoteEnvironmentManager
                 ):
        self.input_df = input_df
        self._entity_label = entity_label
        self._num_permutations = num_permutations
        self._threshold = threshold
        self._minhashes = None
        self._lsh = None
        self._env_manager = env_manager

    @property
    def input_df(self) -> pd.DataFrame:
        '''
        The dataframe containing the chain of shingles.
        '''
        return self._input_df

    @input_df.setter
    def input_df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('input_df must be a pandas DataFrame')
        self._input_df = value

    @property
    def entity_label(self) -> str:
        '''
        The text column to preprocess.
        '''
        return self._entity_label

    @property
    def num_permutations(self) -> int:
        '''
        The number of permutations for constructing the LSH.
        '''
        return self._num_permutations

    @property
    def threshold(self) -> float:
        '''
        The threshold of the Jaccard similarity for
        bucketize the entries in the LSH.
        '''
        return self._threshold

    def _hash_function(self, element: bytes) -> int:
        '''
        Generate a 32-bit integer hash of the input string using the xxhash algorithm.
        '''
        return xxh32_intdigest(element)

    def _create_minhashes(self):
        '''
        Encode the label to be hashed and create a MinHash
        for each entry in the dataframe.
        '''
        input_column = f'shingled_{self._entity_label}_filtered'
        minhashes_input = ([word.encode('utf-8') for word in row]
                           for row in self.input_df[input_column])
        logger.info('Creating all the MinHash. This might take a while.')
        self._minhashes = MinHash.bulk(minhashes_input,
                                       num_perm=self._num_permutations,
                                       hashfunc=self._hash_function)
        self._minhashes = [LeanMinHash(minhash) for minhash in self._minhashes]

class TrainingLSHProcessor(BaseLSHProcessor):
    '''
    The specialized LSH Preprocessor for the training phase.
    It preprocess the input labels, create the MinHashes,
    construct the LSH index and dump it.
    '''

    def _preprocess(self):
        '''
        Discard non necessary entries from the input dataframe
        and split the chain of shingles in a list of strings.
        '''
        input_column = f'shingled_{self._entity_label}_filtered'
        self.input_df = self.input_df[self.input_df['lsh_index'] != -1]
        self.input_df.drop_duplicates(subset='lsh_index', inplace=True)
        self.input_df[input_column] = self.input_df[input_column].str.split(',')
        self.input_df.dropna(inplace=True)

    def _update_lsh_index(self):
        '''
        Update the object LSH with the indexes and the Minashes.

        :param lsh_index: list. The list of indexes for the LSH.
        '''
        lsh_index = self.input_df["lsh_index"].astype('int').items()
        with self._lsh.insertion_session() as session:
            for (_, index), minhash in zip(lsh_index, self._minhashes):
                session.insert(index, minhash)

    def _store_lsh_data(self):
        protocol = self._env_manager.personne_morale_hashtables_metadata.protocol
        storage_options = self._env_manager.personne_morale_hashtables_metadata.storage_options
        file_system = filesystem(protocol, **storage_options)
        with file_system.open(self._env_manager.personne_morale_hashtables_metadata.path,
                              'w') as flux:
            json.dump(self._lsh.metadata, flux)
        storage_options = self._env_manager.storage_options
        output_filename = f"{self._env_manager.personne_morale_hashtables}"
        self._lsh.hashtable_df.to_parquet(output_filename,
                                          engine='pyarrow',
                                          compression='gzip',
                                          storage_options=storage_options)

    def run(self):
        '''
        Perform the entire pipeline for the preparation of the
        LSH index. Store at the end the constructed LSH.
        '''
        logger.info('Preprocessing %s', self._entity_label)
        self._preprocess()
        self._create_minhashes()
        self._lsh = CustomLSH(self._threshold,
                              self._num_permutations)
        logger.info('Updating the LSH index. This might take a while.')
        self._update_lsh_index()
        self._lsh.prepare_data_to_store()
        logger.info('Storing the LSH index.')
        self._store_lsh_data()

class PredictionLSHProcessor(BaseLSHProcessor):
    '''
    The specialized LSH Preprocessor for the prediction phase.
    It preprocess the input labels, create the MinHashes and
    load the LSH index queying each label from it.
    '''

    def _preprocess(self):
        '''
        Split the chain of shingles in a list of strings.
        '''
        input_columns = [f'shingled_{self._entity_label}_filtered',
                         f'shingled_{self._entity_label}_unfiltered']
        self.input_df[input_columns] = self.input_df[input_columns].map(lambda x: x.split(','),
                                                                        na_action='ignore')
        self.input_df.dropna(inplace=True)

    def _get_bytes_hashvalues(self, minhash, hashranges):
        start_range = hashranges[0]
        end_range = hashranges[1]
        return bytes(minhash.hashvalues[start_range:end_range].byteswap().data)

    def _prepare_query_for_signature(self) -> str:
        '''
        Prepare the query to get the candidates given the signatures
        '''
        query_candidates = f"""
        SELECT COMPAUXLIB, shingled_COMPAUXLIB_unfiltered, shingled_COMPAUXLIB_filtered, candidates
        FROM minhashes_table
        LEFT JOIN denomination_hashtable AS hashtable
        ON hashtable.bucket = minhashes_table.bucket AND hashtable.signatures = minhashes_table.signatures"""
        return query_candidates

    def _preprocess_output_dataframe(self):
        '''
        Expand the candidates in the dataframe and replace the NaN
        cases with a dummy value
        '''
        self.input_df['candidates'] = self.input_df['candidates'].str.split(',')
        self.input_df = self.input_df.explode('candidates', ignore_index=True)
        self.input_df.drop_duplicates(subset=['COMPAUXLIB', 'candidates'], inplace=True)
        self.input_df.rename(columns={'candidates': 'candidate_lsh_index'}, inplace=True)
        self.input_df.loc[self.input_df['candidate_lsh_index'] == '',
                          'candidate_lsh_index'] = -1
        self.input_df['candidate_lsh_index'] = self.input_df['candidate_lsh_index'].astype(int)

    def _expand_buckets(self,
                        nb_bands: int,
                        minhashes: list):
        '''
        Unpivot the signatures for each bucket.
        '''
        bucket_colums = [f'{bucket}' for bucket in range(nb_bands)]
        self.input_df[bucket_colums] = minhashes
        self.input_df = self.input_df.melt(id_vars=[column for column in self.input_df.columns
                                                    if column not in bucket_colums],
                                           value_vars=bucket_colums,
                                           var_name='bucket',
                                           value_name='signatures')
        self.input_df['bucket'] = self.input_df['bucket'].astype(int)

    def run(self, connection):
        '''
        Perform the entire pipeline for the querying the
        LSH index. Update the input dataframe with the indexes
        of the LSH containing similar entries to the input ones.
        '''
        self._preprocess()
        self._create_minhashes()
        file_system = filesystem(
            self._env_manager.personne_morale_hashtables_metadata.protocol,
            **self._env_manager.personne_morale_hashtables_metadata.storage_options)
        with file_system.open(
            self._env_manager.personne_morale_hashtables_metadata.path, 'r') as flux:
            metadata = json.load(flux)
        splitted_minhashes = [[self._get_bytes_hashvalues(
            minhash,
            metadata[f'hashrange_{bucket}']
            ) for bucket in range(metadata['nb_bands'])
            ] for minhash in self._minhashes]
        self._expand_buckets(metadata['nb_bands'],
                             splitted_minhashes)
        query_candidates = self._prepare_query_for_signature()
        connection.register('minhashes_table', self.input_df)
        execution = connection.execute(query_candidates)
        self.input_df = execution.df()
        connection.unregister('minhashes_table')
        self.input_df.dropna(subset='candidates', inplace=True)
        self._preprocess_output_dataframe()
