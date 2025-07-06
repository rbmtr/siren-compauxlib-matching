'''
Script for setting up all the configurations necessary to run the project.
'''
import os
import json
import argparse
import logging
import sys
from azure.monitor.opentelemetry import configure_azure_monitor
from environment_manager import LocalEnvironmentManager, RemoteEnvironmentManager

logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)
default_logger = logging.getLogger()
logger = logging.getLogger()

class Startup():
    '''
    Class collecting the methods for setting up environment
    and configuration at the start of the run.
    '''
    __slots__ = (
        "config",
        "args",
        "env_manager",
        "queries",
        "logger",
        "api_name"
        )

    def __init__(self, api_name: str=None):
        self.api_name = api_name
        self.config = None
        self.args = None
        self.env_manager = None
        self.queries = None
        self.logger = None

    def _update_asp_config(self, old_config: dict, new_config: dict):
        '''
        Updating the default config parameters with those
        specific to the deployment environment
        '''
        for key, value in new_config.items():
            if isinstance(value, dict):
                old_config[key] = self._update_asp_config(old_config.get(key, {}), value)
            else:
                old_config[key] = value
        return old_config
    
    def _setup_config(self):
        '''
        Load the config and setup the aspnet environment
        '''
        # Loading the configuration according to the environment
        with open('config/appsetting.json',
                  'r',
                  encoding='utf-8') as config_file:
            self.config = json.load(config_file)
        asp_environment = os.getenv("ASPNETCORE_ENVIRONMENT")
        if asp_environment is not None:
            with open(f"config/appsetting.{asp_environment}.json",
                      "r",
                      encoding='utf-8') as env_config:
                config_asp_environment = json.load(env_config)
            self.config = self._update_asp_config(self.config,
                                                  config_asp_environment)
            # Setting up the name for the logging service
            os.environ["OTEL_SERVICE_NAME"] = asp_environment
            if self.api_name is not None:
                os.environ["OTEL_SERVICE_NAME"] = '.'.join([asp_environment, self.api_name])
        # Setting up the connection string for Azure Application Insight
        if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING") is None:
            if self.config["Azure"]["AZURE_CONNECTION_STRING"] == "":
                logger.error('APPLICATIONINSIGHTS_CONNECTION_STRING should be set either as environment variable or in the config.')
                sys.exit(1)
            os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = self.config["Azure"]["AZURE_CONNECTION_STRING"]

    def _setup_logging(self):
        '''
        Setting up the logger
        '''
        logging.basicConfig(level=self.config['Logging']['LogLevel']['Default'])
        # Setting up Azure logging
        self.logger = logging.getLogger("azure")
        self.logger.setLevel(self.config['Logging']['LogLevel']['Azure'])
        if os.getenv("OTEL_SERVICE_NAME") is not None:
            configure_azure_monitor()
        excluded_logger_names = ["urllib3.connectionpool"]
        for logger_name in excluded_logger_names:
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.WARNING)
        self.logger = logging.getLogger()

    def _setup_parser(self):
        '''
        Parse the arguments from CLI
        '''
        parser = argparse.ArgumentParser(prog='CompAuxLib-SIREN matching')
        parser.add_argument('task',
                            help="""
                            task to perform.
                            Allowed values are train, predict, evaluate_metrics
                            and query_string""",
                            choices=['train',
                                     'predict'])
        parser.add_argument('-e',
                            '--environment',
                            help="""
                            the environment to use. Allowed values are local and
                            remote. Remote environment interact with the Azure blob
                            storage to recover and store data. Default value is
                            local.
                            """,
                            choices=['local', 'remote'],
                            default='local')
        parser.add_argument('-f',
                            '--force_training',
                            help="""
                            defines if the training has to be performed regardless of the
                            presence of the training data in the storage. When forcing
                            the existing resources are overwritten.
                            """,
                            action='store_true')
        parser.add_argument('--fec',
                            help="""
                            the path to the FEC to be treated. It is required only in
                            the prediction phase.
                            """,
                            required=False)
        parser.add_argument('-v',
                            '--version',
                            action='version',
                            version='%(prog)s 0.1')
        self.args = parser.parse_args()

    def _setup_env_manager(self):
        '''
        Setting up the environment manager class
        '''
        self.logger.info('Configuring the environment')
        # Preparing the environment context (either local or remote)
        if self.args.environment == 'local':
            self.env_manager = LocalEnvironmentManager(self.config)
        elif self.args.environment == 'remote':
            self.env_manager = RemoteEnvironmentManager(self.config)
        else:
            self.logger.error('Only local or remote environment are allowed.')
            sys.exit(1)

    def _load_queries(self):
        '''
        Loading the queries in a dictionary
        '''
        # Loading the queries
        with open('queries/queries.json', 'r', encoding='utf-8') as query_file:
            self.queries = json.load(query_file)

    def global_setup(self):
        '''
        Running all the setup steps
        '''
        self._setup_config()
        self._setup_logging()
        self._load_queries()
        self.env_manager = LocalEnvironmentManager(self.config)
        if self.api_name is None:
            self._setup_parser()
            self._setup_env_manager()
