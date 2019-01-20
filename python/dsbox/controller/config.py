import json
import logging
import os


class DsboxConfig(dict):
    '''
    Class for loading and managing DSBox configurations.

    The following variables are defined in D3M configuration files:
    * dataset_schema: File path to datasetDoc.json
    * pipeline_logs_root: Directory for saving pipeline json descriptions
    * executables_root: Directory for saving executables. Note: Probably will not be needed.
    * user_problems_root: Directory for saving user generated problems
    * temp_storage_root: Directory for saving scratch data
    * training_data_root: top-level directory containing the training data
    * test_data_root: top-level directory containing the test data

    An official D3M configuration file can only either have training_data_root or
    test_data_root, not both.

    The following variables are defined in D3M OS environment
    * d3m_run: Run either in 'ta2' for 'ta2ta3' mode (os.environ['D3MRun'])
    * input_root: Top-level directory for all inputs (os.environ['D3MINPUTDIR'])
    * problem_schema: File path to problemDoc.json (os.environ['D3MPROBLEMPATH'])
    * output_root: Top-level directory for all outputs (os.environ['D3MOUTPUTDIR'])
    * local_root: A local-to-host directory used for memory sharing (os.environ['D3MLOCALDIR'])
    * static_root: Directory containing primitives' static fiels (os.environ['D3MSTATICDIR'])
    * cpu: Available CPU units, for example 56.
    * ram: Available memory in GB, for example 15.
    * timeout: Time limit in seconds, for example 3600.

    DSBox variables
    * search_method: pipeline search methods, possible values 'serial', 'parallel', 'random-dimensional', 'bandit', 'multi-bandit'
    * is_multiprocess: if False, then should not spawn subprocesses. Needed for TA3 mode.
    * logs_root: Directory to store logs

    Older unsed varaiables:
    * problem_root: Directory containing problem schema file.

    '''

    OUTPUT_VARIABLES = [
        'pipeline_logs_root',
        'executables_root',
        'user_problems_root',
        'temp_storage_root',
        'logs_root',
        'output_root'
    ]

    INPUT_VARIABLES = [
        'dataset_schema',
        'training_data_root',
        'test_data_root',
        'problem_schema'
    ]

    RESOURCE_VARIABLES = [
        'cpu',
        'ram',
        'timeout'
    ]

    def load(self, filepath):
        self.load_config_json(filepath)
        self.load_d3m_environment()
        self.load_dsbox()
        self._fill_variables()
        print(self)

    def load_ta3(self, *, output_root=''):
        self.load_d3m_environment()
        if output_root is not '':
            self['output_root'] = output_root
        self['pipeline_logs_root'] = os.path.join(self['output_root'], 'pipelines')
        self['executables_root'] = os.path.join(self['output_root'], 'executables')
        self['user_problems_root'] = os.path.join(self['output_root'], 'user_problems')
        self['temp_storage_root'] = os.path.join(self['output_root'], 'supporting_files')
        self.load_dsbox()
        self._fill_variables()

        # TA2TA3 grpc does not work with multi-process
        self['search_method'] = 'serial'

        print(self)

    def load_config_json(self, filepath):
        with open(filepath) as data:
            config = json.load(data)
            self.update(config)

    def load_d3m_environment(self):
        if 'D3MRUN' in os.environ:
            self['d3m_run'] = os.environ['D3MRUN']
        if 'D3MINPUTDIR' in os.environ:
            self['input_root'] = os.environ['D3MINPUTDIR']
        if 'D3MPROBLEMPATH' in os.environ:
            self['problem_schema'] = os.environ['D3MPROBLEMPATH']
        if 'D3MOUTPUTDIR' in os.environ:
            self['output_root'] = os.environ['D3MOUTPUTDIR']
        if 'D3MLOCALDIR' in os.environ:
            self['local_root'] = os.environ['D3MLOCALDIR']
        if 'D3MSTATICDIR' in os.environ:
            self['static_root'] = os.environ['D3MSTATICDIR']
        if 'D3MCPU' in os.environ:
            self['cpu'] = os.environ['D3MCPU']
        if 'D3MRAM' in os.environ:
            self['ram'] = os.environ['D3MRAM']
        if 'D3MTIMEOUT' in os.environ:
            self['timeout'] = os.environ['D3MTIMEOUT']

    def load_dsbox(self):
        self.load_logging()
        if 'search_method' not in self:
            self['search_method'] = 'parallel'
            # self['search_method'] = 'serial'

    def load_logging(self):
        '''
        Config logging level.

        Example:
            export DSBOX_LOGGING_LEVEL="dsbox=WARNING:dsbox.controller=DEBUG:console_logging_level=WARNING:file_logging_level=DEBUG"

            All classes under 'dsbox*' hierarchy log at WARNING level, except 'dsbox.controller*' log at DEBUG level.
            Console log at WARNING level. File log at DEBUG level
        '''

        self['file_formatter'] = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
        self['file_logging_level'] = logging.INFO
        self['log_filename'] = 'dsbox.log'
        self['console_formatter'] = "[%(levelname)s] - %(name)s - %(message)s"
        self['console_logging_level'] = logging.INFO

        LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        min_level = logging.WARNING
        if 'DSBOX_LOGGING_LEVEL' in os.environ:
            for assignment in os.environ['DSBOX_LOGGING_LEVEL'].split(':'):
                try:
                    strings = assignment.split('=')
                    name = strings[0]
                    level = strings[1]
                    if level in LEVELS:
                        level = eval('logging.'+level)
                    else:
                        level = int(level)

                    if name in ('file_logging_level', 'console_logging_level'):
                        self[name] = level
                        print(f'Set logging handler {name} to {level}')
                    else:
                        print(f'Set logger "{name}" level to {level}')
                        logging.getLogger(name).setLevel(level)

                    if level < min_level:
                        min_level = level

                except ValueError:
                    print(f'[ERROR] Skipping logging assignment: {assignment}')

        min_level = min(min_level, self['file_logging_level'], self['console_logging_level'])
        self['root_logger_level'] = min_level
        print(f'Root logger level {min_level}')

    def _fill_variables(self):
        if 'output_root' not in self:
            self['output_root'] = os.path.split(self['executables_root'])[0]
        if 'logs_root' not in self:
            self['logs_root'] = os.path.join(self['temp_storage_root'], 'logs')

    def map_output_variables(self, output_prefix, org_output_prefix='/output/'):
        '''
        Replace string prefix for output oriented variables.

        This is useful for using d3m docker-based configuration files in local environment.
        '''
        for key in self.OUTPUT_VARIABLES:
            self._map_variable(key, output_prefix, org_output_prefix)

    def map_input_variables(self, input_prefix, org_input_prefix='/input/'):
        '''
        Replace string prefix for input oriented variables.

        This is useful for using d3m docker-based configuration files in local environment.
        '''
        for key in self.INPUT_VARIABLES:
            self._map_variable(key, input_prefix, org_input_prefix)

    def _map_variables(self, key, prefix, org_prefix):
        if key in self and self[key].startswith(org_prefix):
            suffix = self[key].split(org_prefix, 1)[1]
            self[key] = os.path.join(prefix, suffix)
