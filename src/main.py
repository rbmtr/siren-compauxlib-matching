'''
Main script for the reconciliation of the COMPAUXLIB with the SIREN.
'''
#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import sys
from upath import UPath
import duckdb
from preprocessor import TrainingPreprocessor, PredictionPreprocessor
from lsh_processor import TrainingLSHProcessor, PredictionLSHProcessor
from estimator import Estimator
from startup import Startup

startup = Startup()
startup.global_setup()
config = startup.config
logger = startup.logger
args = startup.args
env_manager = startup.env_manager
queries = startup.queries

DUCKDB_DB_IN_MEMORY_NAME = "compauxlib"
QUERY_AZURE_SECRET = queries['create_azure_secret'
                                ].format(
                                    azure_storage_account=config['Azure'
                                                                ]['AZURE_STORAGE_ACCOUNT'])

if __name__ == '__main__':
    if args.task == 'train':
        if args.force_training:
            logger.info("""Forcing the execution of the training phase.
                        The files already existing will be overwritten.""")
        else:
            logger.info('Checking presence of lsh_index table')
            if env_manager.personne_morale_lsh_index_path.exists():
                logger.info('Preprocessed tables already existing. Finishing the task.')
                sys.exit(0)
            elif not env_manager.personne_morale_path.exists():
                logger.error('The required table personne_morale is not available.')
                sys.exit(1)
        logger.info('Starting training task')
        logger.info('Recovering the the informations for the personne morale')
        query_personne_morale = queries['get_siren_table'
                                        ].format(input_table=
                                                 env_manager.personne_morale_path)
        with duckdb.connect(f":memory:{DUCKDB_DB_IN_MEMORY_NAME}") as db_connection:
            if args.environment == 'remote':
                db_connection.execute(QUERY_AZURE_SECRET)
            df_personne_morale = db_connection.sql(query_personne_morale).df()
        logger.info('Perfoming preprocessing pipeline')
        denomination_entities = config['model_parameters'
                                       ]['unite_legale_denominations'
                                         ] + config['model_parameters'
                                                    ]['etablissement_denominations'
                                                      ]
        preprocessor = TrainingPreprocessor(df_personne_morale,
                                            'denomination',
                                            env_manager)
        df_siren_preprocessed = preprocessor.run(config['model_parameters']['legal_entities'],
                                                 denomination_entities)
        del df_personne_morale
        logger.info('Pushing the preprocessed table to the database')
        file_to_store = env_manager.personne_morale_lsh_index_path
        df_siren_preprocessed.to_parquet(file_to_store,
                                         engine='pyarrow',
                                         compression='gzip',
                                         storage_options=file_to_store.storage_options)
        logger.info('Starting the construction of the LSH index')
        lsh_processor = TrainingLSHProcessor(df_siren_preprocessed,
                                             'denomination',
                                             config['model_parameters'
                                                    ]['minhash_num_permutations'],
                                             config['model_parameters'
                                                    ]['lsh_threshold'],
                                             env_manager)
        lsh_processor.run()
    elif args.task == 'predict':
        if args.fec is None:
            logger.error('The argument --fec is mandatory in the prediction phase.')
            sys.exit(1)
        data_folder = UPath(config['local_io']['path_to_data'])    
        db_destination = UPath(data_folder,
                               config['io_filename']['lsh_db'])    
        with duckdb.connect(db_destination) as db_connection:    
            if args.environment == 'remote':
                db_connection.execute(QUERY_AZURE_SECRET)
            lfec_database_path = UPath(args.fec)
            db_connection.execute(
                f"ATTACH '{lfec_database_path}' AS {config['sqlite_ediag_alias']} (TYPE SQLITE)")
            logger.info('Recovering COMPAUXLIB for %s', lfec_database_path.stem)
            df_fec = db_connection.sql(queries['get_compauxlib_from_ediag']).df()
            if (df_fec['COMPAUXLIB'].nunique() == 0) | all(df_fec['COMPAUXLIB'].unique() == ''):
                logger.warning('There are no entries for COMPAUXLIB')
                sys.exit(0)
            logger.info('Perfoming preprocessing pipeline')
            preprocessor = PredictionPreprocessor(df_fec,
                                                  'COMPAUXLIB',
                                                  env_manager)
            df_fec = preprocessor.run(config['model_parameters']['legal_entities'])
            logger.info('Recovering SIREN candidates')
            lsh_processor = PredictionLSHProcessor(df_fec,
                                                   'COMPAUXLIB',
                                                   config['model_parameters'
                                                          ]['minhash_num_permutations'],
                                                   config['model_parameters'
                                                          ]['lsh_threshold'],
                                                   env_manager)
            lsh_processor.run(db_connection)
            logger.info('Preparing output table')
            lsh_indexes = lsh_processor.input_df['candidate_lsh_index'].unique()
            lsh_index_to_string = ', '.join([str(index) for index in lsh_indexes])
            lsh_table_name = env_manager.personne_morale_lsh_index_path
            query_siren = "SELECT * FROM lsh_index_siren WHERE lsh_index IN ({indexes}) AND lsh_index != -1".format(indexes=lsh_index_to_string)
            df_query = db_connection.sql(query_siren).df()
            df_query = df_query[df_query['lsh_index'] != -1].reset_index(drop=True)
            siren_to_query = list(set(i[:9] for i in df_query['siret'].unique()))
            # Getting the siren from the table RAPPORT_EDIAG
            query_result = db_connection.sql(queries["get_fec_from_ediag"
                                                     ].format(sqlite_db_alias=config[
                                                         "sqlite_ediag_alias"])).df()
            fec_siren = query_result['FEC'].values[0][:9]
            siren_to_query.append(fec_siren)
            query_personne_morale = queries['get_siren_table_prediction']
            query_personne_morale = ' '.join([query_personne_morale, 'WHERE siren IN ({sirens})'])
            query_personne_morale = query_personne_morale.format(sirens = '?, ' * len(siren_to_query))
            df_personne_morale = db_connection.execute(query_personne_morale, siren_to_query).df()
            df_fec_w_candidates = lsh_processor.input_df.merge(df_query,
                                                               how='left',
                                                               left_on='candidate_lsh_index',
                                                               right_on='lsh_index')
            df_fec_w_candidates.drop(columns='lsh_index', inplace=True)
            df_fec_w_candidates.loc[df_fec_w_candidates['siret'].isna(), 'siret'] = ''
            # Getting the coordinates of the siren
            fec_coordinates = df_personne_morale.loc[df_personne_morale['siren'] == fec_siren,
                                                     ['siret',
                                                      'AbscisseEtablissement',
                                                      'OrdonneeEtablissement']]
            if len(fec_coordinates) == 0:
                logger.warning('Coordinates for FEC sites not found')
            fec_coordinates.rename(columns={'AbscisseEtablissement': 'Abscisse',
                                            'OrdonneeEtablissement': 'Ordonnee'},
                                   inplace=True)
            estimator = Estimator(df_fec_w_candidates,
                                  df_personne_morale,
                                  config['model_parameters']['unite_legale_denominations'],
                                  config['model_parameters']['etablissement_denominations'],
                                  config['model_parameters']['lsh_threshold'],
                                  config['model_parameters']['nb_rank'],
                                  fec_coordinates)
            estimator.run(config['model_parameters']['nb_final_candidates'])
            df_fec_w_candidates = estimator.input_df
            df_to_propose = df_fec_w_candidates[df_fec_w_candidates['validated']
                                                ].reset_index(drop=True)
            df_to_propose.rename(columns={'jaccard_distance_unfiltered': 'jaccard_distance'},
                                 inplace=True)
            # Loading output schema
            with open('schema/output_schema.json', 'r', encoding='utf-8') as schema_file:
                output_schema = json.load(schema_file)
            df_debug = df_fec_w_candidates[output_schema['compauxlib_siren_debug'].keys()]
            list_of_table = db_connection.query(queries["get_ediag_table"
                                                        ].format(sqlite_db_alias=config[
                                                            "sqlite_ediag_alias"])).df()
            if 'compauxlib_siren' not in list_of_table['table_name']:
                for table in ['compauxlib_siren', 'compauxlib_siren_debug']:
                    table_columns = ', '.join([str(column) + ' ' + str(types) for column,
                                               types in output_schema[table].items()])
                    create_table = queries[f'create_{table}_table'].format(
                        list_of_variables=table_columns)
                    db_connection.execute(create_table)
            for table in ['compauxlib_siren', 'compauxlib_siren_debug']:
                push_table = queries[f'update_{table}_table'].format(
                    list_of_variables=', '.join(output_schema[table].keys()))
                db_connection.sql(push_table)
            db_connection.execute(f"DETACH {config['sqlite_ediag_alias']}")
    else:
        raise ValueError('''
                         Invalid argument for action. Valid choises are train,
                         predict.
                         ''')
