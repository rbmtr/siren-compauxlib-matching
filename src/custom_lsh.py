'''
Custom LSH script.

The class is a child of the MinHashLSH class in datasketch. The main reason for the class
is to easily extract the hashtables and the metadata related and to put the information
in a DataFrame in order to be stored later on in some tabular format.
Contrary to the original class, CustomLSH does not give the possibility to use an hashfunction
for the mapping of a MinHash band to a bucket, simply because it would have mean including
the pickling of the hashfunction for later use. In a future implementation we could
include the use of predefined hashfunctions that can be included in the main project.

The inputs are the threshold for the Jaccard similarity and the number of permutations.

The class requires "pandas" and "datasketch" as external package.
'''
import pandas as pd
from datasketch.lsh import MinHashLSH
from datasketch.storage import DictSetStorage


class CustomLSH(MinHashLSH):
    '''
    The customized MinHashLSH class for performing the
    Locality Sensitive Hashing using MinHash and extracting the
    hashtables.
    '''

    __slots__ = (
        "hashtable_df",
        "_metadata"
        )

    def __init__(self,
                 threshold,
                 num_permutations):
        super().__init__(threshold=threshold,
                         num_perm=num_permutations)
        if self.hashfunc is not None:
            raise ValueError('hashfunction not yet implemented.')
        self.hashtable_df = None
        self.metadata = None
    
    def _hashtable_as_dataframe(self,
                                hashtable: DictSetStorage,
                                bucket: int) -> pd.DataFrame:
        '''
        Define a new pandas DataFrame based on the values of
        the hashtable in input.
        
        :param hashtable: DictSetStorage. The dictionary containing the
        MinHash signatures as key and the LSH indexes as values.
        :param bucket: int. The id of the bucket.
        
        return pandas DataFrame. A DataFrame with a column for the
        MinHash signatures and one with the LSH indexes.
        '''
        signatures = list(hashtable.keys())
        candidates = [','.join(str(candidate) for candidate in hashtable.get(signature))
                      for signature in signatures]
        dataframe_hashtable = pd.DataFrame({'signatures': signatures, 'candidates': candidates})
        dataframe_hashtable['bucket'] = bucket
        return dataframe_hashtable

    def _prepare_metadata(self):
        '''
        Construct a dictionary containing some informations necessary for
        using the hashtables at prediction time.
        '''
        self.metadata = {'nb_permutations': self.h,
                         'nb_bands': self.b}
        self.metadata.update({f'hashrange_{table}': self.hashranges[table]
                              for table in range(self.b)})

    def prepare_data_to_store(self):
        '''
        Construct the metadata and the dataframes for the
        hashtables
        '''
        self._prepare_metadata()
        list_of_hashtables_df = [self._hashtable_as_dataframe(self.hashtables[bucket], bucket) for bucket in range(self.b)]
        self.hashtable_df = pd.concat(list_of_hashtables_df)
