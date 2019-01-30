import json
import logging
import os
import typing

import d3m.metadata.base as metadata_base
from d3m.metadata.problem import parse_problem_description


class DsboxConfig:
    '''
    Class for loading and managing DSBox configurations.

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

    D3M output directory structure:
    * pipelines_ranked (pipelines_ranked_dir) - a directory with ranked pipelines to be evaluated, named <pipeline id>.json; these files should have additional field pipeline_rank
    * pipelines_scored (pipelines_scored_dir) - a directory with successfully scored pipelines during the search, named <pipeline id>.json
    * pipelines_searched (pipelines_searched_dir) - a directory of full pipelines which have not been scored or ranked for any reason, named <pipeline id>.json
    * subpipelines (subpipelines_dir) - a directory with any subpipelines referenced from pipelines in pipelines_* directories, named <pipeline id>.json
    * pipeline_runs (pipeline_runs_dir) - a directory with pipeline run records in YAML format, multiple can be stored in the same file, named <pipeline run id>.yml
    * additional_inputs (additional_inputs_dir) - a directory where TA2 system can store any additional datasets to be provided during training and testing to their pipelines; each dataset should be provided in a sub-directory in a D3M dataset format; all datasets here should have a unique ID; in the case that additional datasets are provided, TA2 should output also pipeline run documents for their ranked pipelines because those pipeline run documents contain information how to map these additional inputs to pipeline inputs

    DSBox output directory root is
    * dsbox_output_root: os.path.join(os.environ['D3MINPUTDIR'], 'dsbox').

    DSBox output directgory structure under dsbox_output_root:
    * logs (log_dir): directory for logging files
    * logs/dfs (dfs_log_dir): directory for detailed dataframe logging

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

    DSBox variables
    * search_method: pipeline search methods, possible values 'serial', 'parallel', 'random-dimensional', 'bandit', 'multi-bandit'
    * is_multiprocess: if False, then should not spawn subprocesses. Needed for TA3 mode.
    * logs_root: Directory to store logs
    * timeout_search: Timeout for search part. Typically equal to timeout less 2 minutes

    Older unsed varaiables:
    * problem_root: Directory containing problem schema file.

    '''

    def __init__(self):
        # D3M environment variables
        self.d3m_run = None
        self.input_root = None
        self.problem_schema = None
        self.output_root = None
        self.local_root = None
        self.static_root = None
        self.cpu = None
        self.ram = None
        self.timeout: int = None

        # D3M output directories
        self.pipelines_ranked_dir = None
        self.pipelines_scored_dir = None
        self.pipelines_searched_dir = None
        self.subpipelines_dir = None
        self.pipeline_runs_dir = None
        self.additional_inputs_dir = None

        # DSBox output directories
        self.dsbox_output_root = None
        self.log_dir = None
        self.dfs_log_dir = None

        # DSBox search
        self.search_method = None
        self.is_multiprocess = None
        self.timeout_search: int = None

        # DSBox logging
        self.file_formatter = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
        self.file_logging_level = logging.INFO
        self.log_filename = 'dsbox.log'
        self.console_formatter = "[%(levelname)s] - %(name)s - %(message)s"
        self.console_logging_level = logging.INFO
        self.root_logger_level = min(self.file_logging_level, self.console_logging_level)

        # ==== Derived variables
        # json dict
        self.problem_doc: typing.Dict = {}
        # parsed problem
        self.problem: typing.Dict = {}
        # Should use self.problem_doc
        self.problem_metadata = None

        # File path to datasetDoc.json file
        self.dataset_schema_file: str = ''
        # json dict
        self.dataset_doc: typing.Dict = {}

        # All datasets under the self.input_root directory
        self._all_datasets: typing.Dict = {}

    def load(self):
        self._load_d3m_environment()
        self._load_dsbox()
        self._setup()

    # def load_ta3(self, *, output_root=''):
    #     self.load_d3m_environment()
    #     if output_root is not '':
    #         self.output_root = output_root
    #     self['pipeline_logs_root'] = os.path.join(self.output_root, 'pipelines')
    #     self['executables_root'] = os.path.join(self.output_root, 'executables')
    #     self['user_problems_root'] = os.path.join(self.output_root, 'user_problems')
    #     self['temp_storage_root'] = os.path.join(self.output_root, 'supporting_files')
    #     self.load_dsbox()

    #     # TA2TA3 grpc does not work with multi-process
    #     self['search_method'] = 'serial'

    #     print(self)

    # def load_config_json(self, filepath):
    #     with open(filepath) as data:
    #         config = json.load(data)
    #         self.update(config)

    def _load_d3m_environment(self):
        self.d3m_run = os.environ['D3MRUN']
        self.input_root = os.environ['D3MINPUTDIR']
        self.output_root = os.environ['D3MOUTPUTDIR']
        self.local_root = os.environ['D3MLOCALDIR']
        self.static_root = os.environ['D3MSTATICDIR']
        self.problem_schema = os.environ['D3MPROBLEMPATH']
        self.cpu = int(os.environ['D3MCPU'])
        self.ram = os.environ['D3MRAM']
        self.timeout = int(os.environ['D3MTIMEOUT'])

    def _setup(self):
        self._define_create_output_dirs()
        self._logger = logging.getLogger(__name__)
        self._all_datasets = self._find_dataset_docs(self.input_root)
        self._load_problem()

        self.timeout_search = self.timeout - 120

    def _load_problem(self):
        with open(os.path.abspath(self.problem_schema)) as file:
            self.problem_doc = json.load(file)
        self.problem = parse_problem_description(os.path.abspath(self.problem_schema))
        self.problem_metadata = metadata_base.Metadata(self.problem_doc)

        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']

        dataset_ids = [obj['datasetID'] for obj in self.problem_doc['inputs']['data']]
        if len(dataset_ids) > 1:
            self._logger.warning(f"ProblemDoc specifies more than one dataset id: {dataset_ids}")
        self.dataset_schema_file = self._all_datasets[dataset_ids[0]]
        with open(self.dataset_schema_file, 'r') as dataset_description_file:
            self.dataset_doc = json.load(dataset_description_file)


    def _find_dataset_docs(self, datasets_dir):
        '''
        Find all datasetDoc.json files under the input root directory.
        '''

        datasets: typing.Dict[str, str] = {}

        for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
            dirpath = os.path.abspath(os.path.join(datasets_dir, dirpath))

            if 'datasetDoc.json' in filenames:
                # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
                # exists in raw data filename).
                dirnames[:] = []

                dataset_path = os.path.join(dirpath, 'datasetDoc.json')

                try:
                    with open(dataset_path, 'r') as dataset_file:
                        dataset_doc = json.load(dataset_file)

                    dataset_id = dataset_doc['about']['datasetID']

                    if dataset_id in datasets:
                        self._logger.warning(
                            "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                                'dataset_id': dataset_id,
                                'dataset': dataset_path,
                                'old_dataset': datasets[dataset_id],
                            },
                        )
                    else:
                        datasets[dataset_id] = dataset_path

                except (ValueError, KeyError):
                    self._logger.exception(
                        "Unable to read dataset '%(dataset)s'.", {
                            'dataset': dataset_path,
                        },
                    )
        return datasets

    def _define_create_output_dirs(self):
        # D3M output directories
        self.pipelines_ranked_dir = os.path.join(self.output_root, 'pipelines_ranked')
        self.pipelines_scored_dir = os.path.join(self.output_root, 'pipelines_scored')
        self.pipelines_searched_dir = os.path.join(self.output_root, 'pipelines_searched')
        self.subpipelines_dir = os.path.join(self.output_root, 'subpipelines')
        self.pipeline_runs_dir = os.path.join(self.output_root, 'pipeline_runs')
        self.additional_inputs_dir = os.path.join(self.output_root, 'additional_inputs')

        # DSBox directories
        self.dsbox_output_root = os.path.join(self.output_root, 'dsbox')
        self.log_dir = os.path.join(self.dsbox_output_root, 'logs')
        self.dfs_log_dir = os.path.join(self.log_dir, 'dfs')

        os.makedirs(self.output_root, exist_ok=True)
        for directory in [
                self.pipelines_ranked_dir, self.pipelines_scored_dir,
                self.pipelines_searched_dir, self.subpipelines_dir, self.pipeline_runs_dir,
                self.additional_inputs_dir,
                self.dsbox_output_root, self.log_dir, self.dfs_log_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

    def _load_dsbox(self):
        self._load_logging()
        self.search_method = 'serial'

    def _load_logging(self):
        '''
        Config logging level.

        Example:
            export DSBOX_LOGGING_LEVEL="dsbox=WARNING:dsbox.controller=DEBUG:console_logging_level=WARNING:file_logging_level=DEBUG"

            All classes under 'dsbox*' hierarchy log at WARNING level, except 'dsbox.controller*' log at DEBUG level.
            Console handler at WARNING level. File handler at DEBUG level
        '''

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

                    if name == 'file_logging_level':
                        self.file_logging_level = level
                        print(f'Set logging handler {name} to {level}')
                    elif name == 'console_logging_level':
                        self.console_logging_level = level
                        print(f'Set logging handler {name} to {level}')
                    else:
                        print(f'Set logger "{name}" level to {level}')
                        logging.getLogger(name).setLevel(level)

                    if level < min_level:
                        min_level = level

                except ValueError:
                    print(f'[ERROR] Skipping logging assignment: {assignment}')

        min_level = min(min_level, self.file_logging_level, self.console_logging_level)
        self.root_logger_level = min_level
        print(f'Root logger level {min_level}')

    # def map_output_variables(self, output_prefix, org_output_prefix='/output/'):
    #     '''
    #     Replace string prefix for output oriented variables.

    #     This is useful for using d3m docker-based configuration files in local environment.
    #     '''
    #     for key in self.OUTPUT_VARIABLES:
    #         self._map_variable(key, output_prefix, org_output_prefix)

    # def map_input_variables(self, input_prefix, org_input_prefix='/input/'):
    #     '''
    #     Replace string prefix for input oriented variables.

    #     This is useful for using d3m docker-based configuration files in local environment.
    #     '''
    #     for key in self.INPUT_VARIABLES:
    #         self._map_variable(key, input_prefix, org_input_prefix)

    # def _map_variables(self, key, prefix, org_prefix):
    #     if key in self and self[key].startswith(org_prefix):
    #         suffix = self[key].split(org_prefix, 1)[1]
    #         self[key] = os.path.join(prefix, suffix)
