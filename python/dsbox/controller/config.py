import io
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
    * input_dir: Top-level directory for all inputs (os.environ['D3MINPUTDIR'])
    * problem_schema: File path to problemDoc.json (os.environ['D3MPROBLEMPATH'])
    * output_dir: Top-level directory for all outputs (os.environ['D3MOUTPUTDIR'])
    * local_dir: A local-to-host directory used for memory sharing (os.environ['D3MLOCALDIR'])
    * static_dir: Directory containing primitives' static fiels (os.environ['D3MSTATICDIR'])
    * cpu: Available CPU units, for example 56.
    * ram: Available memory in GB, for example 15.
    * timeout: Time limit in seconds, for example 3600.

    D3M output directory structure:
    * pipelines_ranked (pipelines_ranked_dir) - a directory with ranked pipelines to be
      evaluated, named <pipeline id>.json; these files should have additional field
      pipeline_rank
    * pipelines_scored (pipelines_scored_dir) - a directory with successfully scored
      pipelines during the search, named <pipeline id>.json
    * pipelines_searched (pipelines_searched_dir) - a directory of full pipelines which
      have not been scored or ranked for any reason, named <pipeline id>.json
    * subpipelines (subpipelines_dir) - a directory with any subpipelines referenced from
      pipelines in pipelines_* directories, named <pipeline id>.json
    * pipeline_runs (pipeline_runs_dir) - a directory with pipeline run records in YAML
      format, multiple can be stored in the same file, named <pipeline run id>.yml
    * additional_inputs (additional_inputs_dir) - a directory where TA2 system can store
      any additional datasets to be provided during training and testing to their
      pipelines; each dataset should be provided in a sub-directory in a D3M dataset
      format; all datasets here should have a unique ID; in the case that additional
      datasets are provided, TA2 should output also pipeline run documents for their
      ranked pipelines because those pipeline run documents contain information how to map
      these additional inputs to pipeline inputs

    DSBox output directory root is
    * dsbox_output_dir: os.environ['D3MINPUTDIR']

    DSBox output directory structure under dsbox_output_dir:
    * pipelines_fitted (pipelines_fitted_dir): directory for storing fitted pipelines
    * logs (log_dir): directory for logging files
    * logs/dfs (dfs_log_dir): directory for detailed dataframe logging

    DSBox variables
    * search_method: pipeline search methods, possible values 'serial', 'parallel', 'random-dimensional', 'bandit', 'multi-bandit'
    * timeout_search: Timeout for search part. Typically equal to timeout less 120 seconds

    '''

    def __init__(self):
        # D3M environment variables
        self.d3m_run: str = ''
        self.input_dir: str = ''
        self.problem_schema: str = ''
        self.output_dir: str = ''
        self.local_dir: str = ''
        self.static_dir: str = ''
        self.cpu: int = 0
        self.ram: str = ''
        self._timeout: int = 0

        # D3M output directories
        self.pipelines_ranked_dir: str = ''
        self.pipelines_scored_dir: str = ''
        self.pipelines_searched_dir: str = ''
        self.subpipelines_dir: str = ''
        self.pipeline_runs_dir: str = ''
        self.additional_inputs_dir: str = ''

        # DSBox output directories
        self.dsbox_output_dir: str = ''
        self.log_dir: str = ''
        self.dfs_log_dir: str = ''

        # DSBox search
        self.search_method = 'serial'
        self.serial_search_iterations = 50
        self.timeout_search: int = 0

        # DSBox logging
        self.file_formatter = "%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
        self.file_logging_level = logging.INFO
        self.log_filename = 'dsbox.log'
        self.console_formatter = "%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
        self.console_logging_level = logging.INFO
        self.root_logger_level = min(self.file_logging_level, self.console_logging_level)

        # ==== Derived variables
        # problem spec in json dict
        self.problem_doc: typing.Dict = {}
        # parsed problem spec (e.g.. string value for task converted to d3m.metadata.problem.TaskType enum)
        self.problem: typing.Dict = {}
        # Should use self.problem_doc
        self.problem_metadata = None

        # List of file path to datasetDoc.json files
        self.dataset_schema_files: typing.List[str] = []
        # json dict
        self.dataset_docs: typing.List[typing.Dict] = []

        # All datasets under the self.input_dir directory
        self._all_datasets: typing.Dict = {}

    @property
    def timeout(self) -> int:
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        self._timeout = value
        self.timeout_search = max(self._timeout - 180, int(self._timeout * 0.8))

    def load(self):
        self._load_d3m_environment()
        self._load_dsbox()
        self._setup()

    def set_problem(self, problem_doc, problem):
        self.problem_doc = problem_doc
        self.problem = problem
        self.problem_metadata = metadata_base.Metadata(self.problem_doc)
        self._load_problem_rest()

    def _load_d3m_environment(self):
        '''
        Get D3M environmental variable values.
        '''
        self.d3m_run = os.environ['D3MRUN']
        self.input_dir = os.environ['D3MINPUTDIR']
        self.output_dir = os.environ['D3MOUTPUTDIR']
        self.local_dir = os.environ['D3MLOCALDIR']
        self.static_dir = os.environ['D3MSTATICDIR']
        self.problem_schema = os.environ['D3MPROBLEMPATH']
        self.cpu = int(os.environ['D3MCPU'])
        self.ram = os.environ['D3MRAM']
        self.timeout = int(os.environ['D3MTIMEOUT'])

    def _load_dsbox(self):
        self._load_logging()
        # self.search_method = 'parallel'
        self.search_method = 'serial'

    def _setup(self):
        self._define_create_output_dirs()
        self._logger = logging.getLogger(__name__)
        self._all_datasets = find_dataset_docs(self.input_dir, self._logger)
        self._load_problem()

        # TA3: Return sooner for TA3
        if 'ta3' in self.d3m_run:
            self.serial_search_iterations = 30

    def _load_problem(self):
        with open(os.path.abspath(self.problem_schema)) as file:
            self.problem_doc = json.load(file)
        self.problem = parse_problem_description(os.path.abspath(self.problem_schema))
        self.problem_metadata = metadata_base.Metadata(self.problem_doc)
        self._load_problem_rest()

    def _load_problem_rest(self) -> None:
        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']

        dataset_ids = [obj['datasetID'] for obj in self.problem_doc['inputs']['data']]
        if len(dataset_ids) > 1:
            self._logger.warning(f"ProblemDoc specifies more than one dataset id: {dataset_ids}")

        for id in dataset_ids:
            if id not in self._all_datasets:
                self._logger.info(f'Available datasets are: {self._all_datasets.keys()}')
                self._logger.error(f'Dataset {id} is not available')
                return

        self.dataset_schema_files = [self._all_datasets[id] for id in dataset_ids]

        for dataset_doc in self.dataset_schema_files:
            with open(dataset_doc, 'r') as dataset_description_file:
                self.dataset_docs.append(json.load(dataset_description_file))

    def _define_create_output_dirs(self):
        '''
        Create output directory structure.
        '''

        # D3M output directories
        self.pipelines_ranked_dir = os.path.join(self.output_dir, 'pipelines_ranked')
        self.pipelines_scored_dir = os.path.join(self.output_dir, 'pipelines_scored')
        self.pipelines_searched_dir = os.path.join(self.output_dir, 'pipelines_searched')
        self.subpipelines_dir = os.path.join(self.output_dir, 'subpipelines')
        self.pipeline_runs_dir = os.path.join(self.output_dir, 'pipeline_runs')
        self.additional_inputs_dir = os.path.join(self.output_dir, 'additional_inputs')
        # DSBox directories
        self.dsbox_output_dir = self.output_dir
        self.pipelines_fitted_dir = os.path.join(self.dsbox_output_dir, 'pipelines_fitted')
        self.log_dir = os.path.join(self.dsbox_output_dir, 'logs')
        self.dfs_log_dir = os.path.join(self.log_dir, 'dfs')

        os.makedirs(self.output_dir, exist_ok=True)
        for directory in [
                self.pipelines_ranked_dir, self.pipelines_scored_dir,
                self.pipelines_searched_dir, self.subpipelines_dir, self.pipeline_runs_dir,
                self.additional_inputs_dir, self.local_dir,
                self.dsbox_output_dir, self.pipelines_fitted_dir, self.log_dir, self.dfs_log_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

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

    def __str__(self):
        out = io.StringIO('DSBox configuration:')
        print(f'  d3m_run: {self.d3m_run}', file=out)
        print(f'  input_dir: {self.input_dir}', file=out)
        print(f'  problem_schema: {self.problem_schema}', file=out)
        print(f'  output_dir: {self.output_dir}', file=out)
        print(f'  local_dir: {self.local_dir}', file=out)
        print(f'  static_dir: {self.static_dir}', file=out)
        print(f'  cpu: {self.cpu}', file=out)
        print(f'  ram: {self.ram}', file=out)
        print(f'  timeout: {self.timeout}', file=out)
        print(f'  timeout_search: {self.timeout_search}', file=out)
        print(f'  search_method: {self.search_method}', file=out)
        content = out.getvalue()
        out.close()
        return content

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


def find_dataset_docs(datasets_dir, _logger=None):
    '''
    Find all datasetDoc.json files under the input root directory.
    '''
    if not _logger:
        _logger = logging.getLogger(__name__)
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
                    _logger.warning(
                        "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                            'dataset_id': dataset_id,
                            'dataset': dataset_path,
                            'old_dataset': datasets[dataset_id],
                        },
                    )
                else:
                    datasets[dataset_id] = dataset_path

            except (ValueError, KeyError):
                _logger.exception(
                    "Unable to read dataset '%(dataset)s'.", {
                        'dataset': dataset_path,
                    },
                )
    return datasets
