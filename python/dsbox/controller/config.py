import json
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
    * test_data_root: top-level directory containing the training data

    An official D3M configuration file can only either have training_data_root or
    test_data_root, not both.

    The following variables are defined in OS environment
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
    * logs_root: Directory to store logs

    Older unsed varaiables:
    * problem_root: Directory containing problem schema file.

    '''

    OUTPUT_VARIABLES = [
        'pipeline_logs_root',
        'executables_root',
        'user_problems_root',
        'temp_storage_root',
        'output_root'
    ]

    INPUT_VARIABLES = [
        'dataset_schema',
        'training_data_root',
        'test_data_root',
        'problem_schema'
        ]

    def load(self, filepath):
        self.load_config_json(filepath)
        self.load_d3m_environment()
        self._fill_variables()
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
