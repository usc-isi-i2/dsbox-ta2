import io
import json
import logging
import os
import time
import typing

from d3m.metadata.problem import Problem
from d3m.metadata.pipeline import Pipeline


class RuntimeSetting:
    '''
    Class for storing information needed for Runtime
    '''
    def __init__(self, *, volumes_dir: str = None, scratch_dir: str = None, log_dir: str = None):
        self.volumes_dir = volumes_dir
        self.scratch_dir = scratch_dir
        self.log_dir = log_dir


class DsboxConfig:
    '''
    Class for loading and managing DSBox configurations.

    The following variables are defined in D3M OS environment
    * d3m_run: valid values are 'ta2' or 'ta2ta3' (os.environ['D3MRun'])
    * deprecated: d3m_context: values are 'TESTING', 'EVALUATION', 'PRODUCTION' (os.environ['D3MCONTEXT'])
    * input_dir: Top-level directory for all inputs (os.environ['D3MINPUTDIR'])
    * problem_schema: File path to problemDoc.json (os.environ['D3MPROBLEMPATH'])
    * output_dir: Top-level directory for all outputs (os.environ['D3MOUTPUTDIR'])
    * local_dir: A local-to-host directory used for memory sharing (os.environ['D3MLOCALDIR'])
    * static_dir: Directory containing primitives' static fiels (os.environ['D3MSTATICDIR'])
    * cpu: Available CPU units, for example 56.
    * ram: Available memory in GB, for example 15.
    * timeout: Time limit in seconds, for example 3600. This property can be set either
      through the environment variable D3MTIMEOUT (in second units), or through
      SearchSolutionRequest time_bound_search field (in minute units). The
      SearchSolutionRequest field takes precedence.

    D3M output directory structure:
    * pipelines_ranked (pipelines_ranked_dir) - a directory with ranked pipelines to be
      evaluated, named <pipeline id>.json; Each json file should have a corresponding
      <pipeline id>.rank file
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
    * pipelines_failed (pipelines_failed_dir): directory for storing failed pipelines
    * logs (log_dir): directory for logging files
    * logs/dfs (dfs_log_dir): directory for detailed dataframe logging

    DSBox variables
    * search_method: pipeline search methods, possible values 'serial', 'parallel', 'random-dimensional', 'bandit', 'multi-bandit'
    * timeout_search: Timeout for search part. The remaining time after timeout_search is used for returning results.

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
        # See timeout property
        self._timeout: int = 0

        # D3M output directories
        self.pipelines_ranked_dir: str = ''
        self.pipelines_scored_dir: str = ''
        self.pipelines_searched_dir: str = ''
        self.subpipelines_dir: str = ''
        self.pipeline_runs_dir: str = ''
        self.additional_inputs_dir: str = ''

        self.pipelines_ranked_temp_dir: str = ''

        # == D3M TA3 SearchSolutionsRequest parameters
        # Number of ranked solution to return
        self.rank_solutions_limit: int = 0
        # Time bound on individual pipeline run. Store as seconds (input is minutes).
        self.time_bound_run: int = 0
        # Random seed used to initiate search
        self.random_seed = 0

        # DSBox output directories
        self.dsbox_output_dir: str = ''
        self.dsbox_scratch_dir: str = ''
        self.log_dir: str = ''
        self.dfs_log_dir: str = ''

        # == DSBox search
        self.search_method = 'serial'
        self.serial_search_iterations = 50
        # Should be set using set_start_time() as soon as the search request is received
        self._start_time: float = 0
        # Search time
        self.timeout_search: int = 0

        # DSBox logging
        self.file_formatter = "%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
        self.file_logging_level = logging.INFO
        self.log_filename = 'dsbox.log'
        self.console_formatter = "%(asctime)s [%(levelname)s] %(name)s -- %(message)s"
        self.console_logging_level = logging.INFO
        self.root_logger_level = min(self.file_logging_level, self.console_logging_level)

        # ==== Derived variables
        self.problem: Problem = {}

        # TA3 can directly supply a pipeline. When pipeline is given, problem spec if optional
        self.pipeline: Pipeline = None

        # List of file path to datasetDoc.json files
        self.dataset_schema_files: typing.List[str] = []
        # json dict
        self.dataset_docs: typing.List[typing.Dict] = []

        # All datasets under the self.input_dir directory
        self._all_datasets: typing.Dict = {}

    @property
    def timeout(self) -> int:
        '''
        '''
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        self._timeout = value
        # self.timeout_search = max(self._timeout - 180, int(self._timeout * 0.93))
        # 2019.7.19: add more time for system clean up job
        self.timeout_search = int(self._timeout * 0.93)

    @property
    def start_time(self) -> float:
        '''
        Returns time.perf_counter counter clock in seconds
        '''
        return self._start_time

    def set_start_time(self):
        '''
        Should be called as soon as the search request is made. Should be called by
        TA2Servicer class and ta2_evaluation.py script.
        '''
        self._start_time = time.perf_counter()

    def load(self, ta2ta3_mode: bool = False):
        self._load_d3m_environment(ta2ta3_mode)
        self._load_dsbox()
        self._setup()

    def set_problem(self, problem: Problem):

        if not isinstance(problem, Problem):
            raise ValueError(f"Argument problem must be an instance of Problem: {problem}")

        if 'id' not in problem:
            raise ValueError(f"Problem missing id: {problem}")

        self.problem = problem
        self._load_problem_rest()

    def get_runtime_setting(self) -> RuntimeSetting:
        return RuntimeSetting(
            volumes_dir=self.static_dir,
            scratch_dir=self.dsbox_scratch_dir,
            log_dir=self.log_dir)

    def _load_d3m_environment(self, ta2ta3_mode: bool):
        '''
        Get D3M environmental variable values.
        '''
        for key, value in os.environ.items():
            if 'D3M' in key:
                print(f'{key}={value}')

        self.d3m_run = os.environ['D3MRUN']
        # self.d3m_context = os.environ.get('D3MCONTEXT', default='TEST')
        self.input_dir = os.environ['D3MINPUTDIR']
        self.output_dir = os.environ['D3MOUTPUTDIR']
        self.local_dir = os.environ['D3MLOCALDIR']
        self.static_dir = os.environ['D3MSTATICDIR']
        if 'D3MCPU' in os.environ:
            self.cpu = int(os.environ['D3MCPU'])
        else:
            import multiprocessing
            self.cpu = multiprocessing.cpu_count() - 1
        if self.cpu < 1:
            self.cpu = 1
        if 'D3MRAM' in os.environ:
            self.ram = os.environ['D3MRAM']

        if ta2ta3_mode:
            # Timeout should not be used in ta2ta3_mode. Set to a large number
            self.timeout = 9999999
            print(f'In ta2ta3_mode, set timeout to {self.timeout}')
            if 'D3MPROBLEMPATH' in os.environ:
                self.problem_schema = os.environ['D3MPROBLEMPATH']
        else:
            if 'D3MTIMEOUT' in os.environ:
                self.timeout = int(os.environ['D3MTIMEOUT'])
            else:
                self.timeout = 9999999
                print('D3MTIMEOUT environment variable not defined. Setting to a large value')
            if 'D3MPROBLEMPATH' in os.environ:
                self.problem_schema = os.environ['D3MPROBLEMPATH']
            else:
                print('D3MPROBLEMPATH environment variable not defined.')
        self.datamart_nyu_url = os.environ.get('DATAMART_URL_NYU', default='')
        self.datamart_isi_url = os.environ.get('DATAMART_URL_ISI', default='')

    def _load_dsbox(self):
        self._load_logging()
        if 'DSBOX_SEARCH_METHOD' in os.environ:
            self.search_method = os.environ['DSBOX_SEARCH_METHOD']
        else:
            self.search_method = 'parallel'
            # self.search_method = 'serial'

    def _setup(self):
        self._define_create_output_dirs()
        self._logger = logging.getLogger(__name__)
        self._all_datasets = find_dataset_docs(self.input_dir, self._logger)
        self._load_problem()

        # TA3: Return sooner for TA3
        if 'ta3' in self.d3m_run:
            self.serial_search_iterations = 30

    def _load_problem(self):
        if self.problem_schema == '':
            return
        self.problem = Problem.load('file://' + os.path.abspath(self.problem_schema))
        self._load_problem_rest()

    def _load_problem_rest(self) -> None:
        # updated v2019.11.14: now use task keywords
        # self.task_keywords = self.problem['problem']['task_keywords']
        self.task_type = self.problem['problem']['task_keywords']
        self.task_subtype = self.problem['problem']['task_keywords']

        dataset_ids = [obj['dataset_id'] for obj in self.problem['inputs']]
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
        self.pipelines_ranked_temp_dir = os.path.join(self.output_dir, 'pipelines_ranked_temp')
        self.dsbox_output_dir = self.output_dir

        # For storing fitted pipeline with pickled primitives
        self.pipelines_fitted_dir = os.path.join(self.dsbox_output_dir, 'pipelines_fitted')

        # For stroing failed pipelines
        self.pipelines_failed_dir = os.path.join(self.dsbox_output_dir, 'pipelines_failed')

        # For storing mappings between fitted pipeline and regular pipeline
        self.pipelines_info_dir = os.path.join(self.dsbox_output_dir, 'pipelines_info')

        # For temporay storage
        self.dsbox_scratch_dir = os.path.join(self.dsbox_output_dir, 'scratch')
        self.log_dir = os.path.join(self.dsbox_output_dir, 'logs')
        self.dfs_log_dir = os.path.join(self.log_dir, 'dfs')

        os.makedirs(self.output_dir, exist_ok=True)
        for directory in [
                self.pipelines_ranked_dir, self.pipelines_ranked_temp_dir, self.pipelines_scored_dir,
                self.pipelines_searched_dir, self.subpipelines_dir, self.pipeline_runs_dir,
                self.additional_inputs_dir, self.local_dir,
                self.dsbox_output_dir, self.pipelines_fitted_dir, self.pipelines_failed_dir, self.pipelines_info_dir,
                self.log_dir, self.dfs_log_dir, self.dsbox_scratch_dir]:
            if not os.path.exists(directory):
                os.mkdir(directory)

    def _load_logging(self):
        '''
        Config logging level.

        Example:
            export DSBOX_LOGGING_LEVEL="dsbox=WARNING:dsbox.template.runtime=DEBUG:console_logging_level=WARNING:file_logging_level=DEBUG"

            All classes under 'dsbox*' hierarchy log at WARNING level, except 'dsbox.template.runtime.*' log at DEBUG level.
            Console handler at WARNING level. File handler at DEBUG level.
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
