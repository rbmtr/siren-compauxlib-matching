'''
Environment manager script.

The script defines the classes containing the paths to the different files necessary
for the project. The path are defined as universal_pathlib, so that can be used with minor
modification for both local files and files stored in an azure storage.
The input is a config dictionary containing the strings for the base paths from which
each file path can be constructed.

The class requires "upath" as external package.
'''

from upath import UPath

class EnvironmentManager():
    '''
    The base environment manager class containing the attributes
    to be used for both local and remote cases.
    '''

    __slots__ = (
        "_config",
        "_personne_morale_filename",
        "_personne_morale_lsh_index_table_filename",
        "_personne_morale_lsh_metadata_filename",
        "_personne_morale_lsh_hashtables_filename",
        "_discarded_token_filename",
        "_storage_options"
        )

    def __init__(self, config: dict):
        self._config = config
        self._personne_morale_filename = self._config[
            'io_filename'
            ]['personne_morale_parquet']
        self._personne_morale_lsh_index_table_filename = self._config[
            'io_filename'
            ]['personne_morale_lsh_index_table']
        self._personne_morale_lsh_metadata_filename = self._config[
            'io_filename'
            ]['denomination_lsh_metadata']
        self._personne_morale_lsh_hashtables_filename = self._config[
            'io_filename'
            ]['lsh_db']
        self._discarded_token_filename = self._config['io_filename']['discarded_token']
        self.storage_options = None

    @property
    def config(self) -> dict:
        '''
        The dictionary containing the filenames
        and path.
        '''
        return self._config

    @property
    def personne_morale_filename(self):
        '''
        The name of the table containing the preprocessed
        personne_morale.
        '''
        return self._personne_morale_filename

    @property
    def personne_morale_lsh_index_table_filename(self):
        '''
        The name of the table containing the lsh_index for
        each siren.
        '''
        return self._personne_morale_lsh_index_table_filename

    @property
    def personne_morale_lsh_metadata_filename(self):
        '''
        The name of the metadata for the hashtables
        of denominationUniteLegale.
        '''
        return self._personne_morale_lsh_metadata_filename

    @property
    def personne_morale_lsh_hashtables_filename(self):
        '''
        The name of the hashtables of
        denominationUniteLegale
        '''
        return self._personne_morale_lsh_hashtables_filename

    @property
    def discarded_token_filename(self):
        '''
        The name of the hashtables of
        denominationUniteLegale
        '''
        return self._discarded_token_filename

    @property
    def storage_options(self):
        '''
        The dictionary containing the informations
        for accessing the Azure blob storage.
        It is None if running on local.
        '''
        return self._storage_options

    @storage_options.setter
    def storage_options(self, value):
        if not isinstance(value, dict) and value is not None:
            raise TypeError('storage_options must be a dictionary or None')
        self._storage_options = value

class LocalEnvironmentManager(EnvironmentManager):
    '''
    The specialized environment manager class for running on local files.
    '''
    def __init__(self, config: dict):
        super().__init__(config)
        self._root_path = self._config['local_io']['root_path_to_data']
        self._siren_path = self._config['local_io']['folder_to_siren_data']
        self.base_data_path = UPath(self._root_path,
                                    self._siren_path)
        self.personne_morale_path = UPath(self.base_data_path,
                                          self._personne_morale_filename)
        self.personne_morale_lsh_index_path = UPath(
            self.base_data_path,
            self._personne_morale_lsh_index_table_filename)
        self.personne_morale_hashtables_metadata = UPath(
            self.base_data_path,
            self._personne_morale_lsh_metadata_filename)
        self.personne_morale_hashtables = UPath(self.base_data_path,
                                                self._personne_morale_lsh_hashtables_filename)
        self.discarded_token_path = UPath(self.base_data_path, self._discarded_token_filename)

class RemoteEnvironmentManager(EnvironmentManager):
    '''
    The specialized environment manager class for running on remote files.
    '''
    def __init__(self, config: dict):
        super().__init__(config)
        self._account_name = self._config['Azure']['AZURE_STORAGE_ACCOUNT']
        self._container_name = self._config['Azure']['AZURE_AUTHORIZED_CONTAINER_NAME']
        self._blob_root_name = self._config['Azure']['AZURE_CONTAINER_PATH']
        self.storage_options = {'account_name': self._account_name,
                                'anon': False}
        base_data_string_path = f"az://{self._container_name}/{self._blob_root_name}"
        self.base_data_path = UPath(base_data_string_path, **self.storage_options)
        self.personne_morale_path = self.base_data_path / self._personne_morale_filename
        self.personne_morale_lsh_index_path = self.base_data_path / self._personne_morale_lsh_index_table_filename
        self.personne_morale_hashtables_metadata = self.base_data_path / self._personne_morale_lsh_metadata_filename
        self.personne_morale_hashtables = self.base_data_path / self._personne_morale_lsh_hashtables_filename
        self.discarded_token_path = self.base_data_path / self._discarded_token_filename
