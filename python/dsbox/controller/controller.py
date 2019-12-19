import enum
import json
import logging
import operator
import os
import pathlib
import pickle
import pprint
import random
import shutil
import time
import traceback
import typing
import copy
import datamart
import multiprocessing
import pandas as pd  # type: ignore
from d3m.base import utils as d3m_utils
from d3m.container.dataset import Dataset, D3MDatasetLoader
from d3m.metadata.base import ALL_ELEMENTS
# no more tasktype since d3m core package v2019.11.10
from d3m.metadata.problem import TaskKeyword
from wikifier import wikifier
from datamart_isi.cache.metadata_cache import MetadataCache
from datamart_isi.utilities.download_manager import DownloadManager
from datamart_isi.utilities.timeout import timeout_call
from datamart_isi import config as config_datamart
from datamart_isi import rest

from dsbox.combinatorial_search.TemplateSpaceBaseSearch import TemplateSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import TemplateSpaceParallelBaseSearch
from dsbox.JobManager.usage_monitor import UsageMonitor
# from dsbox.combinatorial_search.BanditDimensionalSearch import BanditDimensionalSearch
# from dsbox.combinatorial_search.MultiBanditSearch import MultiBanditSearch
from dsbox.controller.config import DsboxConfig
from dsbox.schema import ColumnRole, SpecializedProblem
from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.ensemble_tuning import EnsembleTuningPipeline, HorizontalTuningPipeline
from dsbox.template.library import TemplateLibrary
from dsbox.template.template import DSBoxTemplate

# import dsbox.JobManager.mplog as mplog

__all__ = ['Status', 'Controller', 'controller_instance']


# pd.set_option("display.max_rows", 100)

controller_instance = None


class Status(enum.Enum):
    OK = 0
    ERROR = 1
    PROBLEM_NOT_IMPLEMENT = 148


class Controller:
    TIMEOUT = 59  # in minutes

    def __init__(self, development_mode: bool = False, run_single_template_name: str = "", is_ta3=True) -> None:
        global controller_instance
        if controller_instance is None:
            controller_instance = self

        self.development_mode: bool = development_mode
        self.is_ta3 = is_ta3

        self.run_single_template_name = run_single_template_name

        self.config: DsboxConfig = None

        # Problem
        # self.problem: typing.Dict = {}
        # self.task_type: TaskType = None
        # self.task_subtype: TaskSubtype = None
        # self.problem_doc_metadata: Metadata = None
        self.problem_info: dict = {}
        self.specialized_problem: str = SpecializedProblem.NONE
        self.is_privileged_data_problem = False

        # Dataset
        # self.dataset_schema_file: str = ""
        self.train_dataset1: Dataset = None
        self.train_dataset2: typing.List[Dataset] = None
        self.test_dataset1: Dataset = None
        self.test_dataset2: typing.List[Dataset] = None
        self.all_dataset: Dataset = None
        self.ensemble_dataset: Dataset = None
        self.taskSourceType: typing.Set[str] = set()  # str from SEMANTIC_TYPES
        self.extra_primitive: typing.Set[str] = set()

        # hard coded unsplit dataset type
        # TODO: check whether "speech" type should be put into this list or not
        self.data_type_cannot_split = ["graph", "edgeList", "audio"]
        self.task_type_can_split = ["CLASSIFICATION", "REGRESSION", "SEMISUPERVISED_CLASSIFICATION", "SEMISUPERVISED_REGRESSION"]

        # !!! hard code here
        # TODO: add if statement to determine it
        # Turn off for now
        self.do_ensemble_tune = False
        self.do_horizontal_tune = False

        self.report_ensemble: dict = dict()

        # Resource limits
        # self.ram: int = 0  # concurrently ignored
        # self.timeout: int = 0  # in seconds

        # Templates
        if self.run_single_template_name:
            self.template_library = TemplateLibrary(run_single_template=run_single_template_name)
        else:
            self.template_library = TemplateLibrary()
        self.template_list: typing.List[DSBoxTemplate] = []
        self.max_split_times = 1

        # Primitives
        # self.primitive: typing.Dict = d3m.index.search()

        # set random seed, but do not set in TA3 mode (based on request from TA3 developer)
        if not is_ta3:
            random.seed(4676)

        # Output directories
        # self.output_directory: str = '/output/'
        # self.output_pipelines_dir: str = ""
        # self.output_executables_dir: str = ""
        # self.output_supporting_files_dir: str = ""
        # self.output_logs_dir: str = ""
        self._logger = logging.getLogger(__name__)

        self.main_pid: int = os.getpid()

        # Template search method
        self._search_method = None

        # Set sample size (sample used to do wikifier comes from supplied_data)
        self.wikifier_max_len = 100000
        self.wikifier_selection_rate = 0.1
        self.wikifier_default_size = 1000

        # reosurce monitor
        self.resource_monitor = UsageMonitor()
    """
        **********************************************************************
        Private method
        1. _check_and_set_dataset_metadata
#        2. _create_output_directory
        3. _load_schema
        4. _log_init
        5. _log_search_results
        6. _process_pipeline_submission
        7. _run_BanditDimSearch
        8. _run_ParallelBaseSearch
        9. _run_RandomDimSearch
        10.
        **********************************************************************
    """

    def _check_and_set_dataset_metadata(self) -> None:
        """
        Update metadata of dataframe columns based on problem doc specification. Columns
        changed include target columns and privileged data columns.
        """

        # NOTICE: Should follow d3m.runtime.Runtime._mark_columns()

        # Remove suggest target from all columns, to avoid confusion
        for resource_id, resource in self.all_dataset.items():
            indices = self.all_dataset.metadata.list_columns_with_semantic_types(ColumnRole.SUGGESTED_TARGET)
            for column_index in indices:
                selector = (resource_id, ALL_ELEMENTS, column_index)

                column_semantic_types: list = self.all_dataset.metadata.query(selector)['semantic_types']
                column_semantic_types.remove(ColumnRole.SUGGESTED_TARGET)
                self.all_dataset.metadata = self.all_dataset.metadata.update(
                    selector, {'semantic_types': column_semantic_types})
                self._logger.debug(f'Removing suggest target tag for {selector}')

        # Set true target column(s)
        for dataset in self.config.problem['inputs']:
            if 'targets' not in dataset:
                continue
            for info in dataset['targets']:
                selector = (info['resource_id'], ALL_ELEMENTS, info['column_index'])

                self.all_dataset.metadata = self.all_dataset.metadata.add_semantic_type(selector, ColumnRole.TRUE_TARGET)
                self.all_dataset.metadata = self.all_dataset.metadata.add_semantic_type(selector, ColumnRole.TARGET)

                # Add suggested target, because primitives like LUPI still using suggested target
                self.all_dataset.metadata = self.all_dataset.metadata.add_semantic_type(selector, ColumnRole.SUGGESTED_TARGET)

                # Target is not attribute
                self.all_dataset.metadata = self.all_dataset.metadata.remove_semantic_type(selector, ColumnRole.ATTRIBUTE)

                self._logger.debug(f'Adding true target tag for {selector}')

        # Set privileged data columns
        for dataset in self.config.problem['inputs']:
            # kyao 2019-7-24:
            # if 'LL0_acled' in dataset['dataset_id']:
            #     self.specialized_problem = SpecializedProblem.ACLED_LIKE_PROBLEM

            if 'privileged_data' not in dataset:
                continue
            self.specialized_problem = SpecializedProblem.PRIVILEGED_INFORMATION
            self._logger.debug(f'Specialized problem: {self.specialized_problem}')
            for info in dataset['privileged_data']:
                selector = (info['resource_id'], ALL_ELEMENTS, info['column_index'])

                self.all_dataset.metadata = self.all_dataset.metadata.add_semantic_type(selector, ColumnRole.PRIVILEGED_DATA)
                self._logger.debug(f'Adding privileged info tag for {selector}')

        return

    def _load_schema(self, *, is_ta3=False):
        # Problem
        if is_ta3:
            # TA3 init
            # self.problem: typing.Dict = config['problem_parsed']
            # self.problem_doc_metadata = Metadata(config['problem_json'])
            pass
        else:
            # TA2 init
            # self.problem = parse_problem_description(config['problem_schema'])
            # with open(os.path.abspath(config['problem_schema'])) as file:
            #     problem_doc = json.load(file)
            #     self.problem_doc_metadata = Metadata(problem_doc)
            pass

        # self.task_type = self.config.problem['problem']['task_type']
        # self.task_subtype = self.config.problem['problem']['task_subtype']

        # Dataset
        # self.dataset_schema_file = config['dataset_schema']
        # if self.dataset_schema_file.startswith('file://'):
        #     self.dataset_schema_file = self.dataset_schema_file[7:]

        # find the data resources type
        self.taskSourceType = set()  # set the type to be set so that we can ignore the repeat
        # elements
        for doc in self.config.dataset_docs:
            for each_type in doc["dataResources"]:
                self.taskSourceType.add(each_type["resType"])
        self.problem_info["data_type"] = self.taskSourceType
        # !!!!
        # self.saved_pipeline_id = config.get('saved_pipeline_ID', "")
        self.saved_pipeline_id = ""

        for i in range(len(self.config.problem['inputs'])):
            if 'targets' in self.config.problem['inputs'][i]:
                break

        self.problem_info["task_type"] = [x.name for x in self.config.problem['problem']['task_keywords']]
        # example of task_type : 'classification' 'regression'
        self.problem_info["res_id"] = self.config.problem['inputs'][i]['targets'][0]['resource_id']
        self.problem_info["target_index"] = []
        for each in self.config.problem['inputs'][i]['targets']:
            self.problem_info["target_index"].append(each["column_index"])

    def _log_init(self) -> None:
        logging.getLogger('').setLevel(min(logging.getLogger('').level, self.config.root_logger_level))

        file_handler = logging.FileHandler(
            filename=os.path.join(self.config.log_dir, self.config.log_filename),
            mode='w')
        file_handler.setLevel(self.config.file_logging_level)
        file_handler.setFormatter(logging.Formatter(fmt=self.config.file_formatter, datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger('').addHandler(file_handler)

        # Do not add another console handler
        if logging.StreamHandler not in [type(x) for x in logging.getLogger('').handlers]:
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(self.config.console_formatter, datefmt='%m-%d %H:%M:%S'))
            console.setLevel(self.config.console_logging_level)
            logging.getLogger('').addHandler(console)
        else:
            for handler in logging.getLogger('').handlers:
                if type(handler) is logging.StreamHandler:
                    handler.setFormatter(logging.Formatter(self.config.console_formatter, datefmt='%m-%d %H:%M:%S'))
                    handler.setLevel(self.config.console_logging_level)

        self._logger = logging.getLogger(__name__)

        self._logger.info('Top level output directory: %s' % self.config.output_dir)

    def _log_search_results(self, report: typing.Dict[str, typing.Any]):
        # self.report_ensemble['report'] = report
        candidate = report['configuration']
        print("-" * 20)
        print("[INFO] Final Search Results:")
        pprint.pprint(candidate)

        if candidate is None:
            self._logger.error("[ERROR] No candidate found during search!")
            print("[INFO] cross_validation_metrics:", None)
        else:
            self._logger.info("******************\n[INFO] Writing results")
            metric_list = ['training_metrics', 'cross_validation_metrics', 'test_metrics']
            for m in metric_list:
                if m in report and type(report[m]) is list:
                    self._logger.info('{} {} = {}'.format(
                        m, report[m][0]['metric'], report[m][0]['value']))

        # # writing to the disk
        # dataset_name = self.output_executables_dir.rsplit("/", 2)[1]
        # # save_location = os.path.join(self.output_logs_dir, dataset_name + ".txt")
        # save_location = self.output_directory + ".txt"
        #
        # self._logger.info("******************\n[INFO] Saving training results in %s",
        # save_location)
        # metrics = self.config.problem['problem']['performance_metrics']
        # candidate = report['configuration']
        # try:
        #     f = open(save_location, "w+")
        #     f.write(str(metrics) + "\n")
        #
        #     for m in ["training_metrics", "cross_validation_metrics", "test_metrics"]:
        #         if m in candidate.data and candidate.data[m]:
        #             f.write(m + ' ' + str(candidate.data[m][0]['value']) + "\n")
        #     # f.write(str(candidate.data['training_metrics'][0]['value']) + "\n")
        #     # f.write(str(candidate.data['cross_validation_metrics'][0]['value']) + "\n")
        #     # f.write(str(candidate.data['test_metrics'][0]['value']) + "\n")
        #     f.close()
        # except:
        #     self._logger.exception('[ERROR] Save training results Failed!')
        #     raise NotSupportedError(
        #         '[ERROR] Save training results Failed!')

    def _process_pipeline_submission(self) -> None:
        # If no limit then no need to remove any pipelines
        limit = self.config.rank_solutions_limit
        if limit <= 0:
            limit = 20
            return

        ranked_list = []
        directory = pathlib.Path(self.config.pipelines_ranked_dir)
        for rank_file in directory.glob('*.rank'):
            try:
                rank = float(open(directory / rank_file).read())
                ranked_list.append((rank, rank_file))
            except Exception:
                pass
        ranked_list = sorted(ranked_list, key=operator.itemgetter(0))

        # Keep all solutions
        if len(ranked_list) <= limit:
            return

        # Remove pipelines with larger rank values
        for (rank, rank_file) in ranked_list[limit:]:
            (directory / rank_file).with_suffix('.json').unlink()
            (directory / rank_file).with_suffix('.rank').unlink()

    # def _process_pipeline_submission_old(self) -> None:
    #     self._logger.info(f'Moving top 20 pipelines to {self.config.pipelines_ranked_dir}')

    #     # Get list of (rank, pipeline) pairs
    #     pipeline_files = os.listdir(self.config.pipelines_scored_dir)
    #     ranked_list = []
    #     for filename in pipeline_files:
    #         if filename.endswith("json"):
    #             filepath = os.path.join(self.config.pipelines_scored_dir, filename)
    #             with open(filepath) as f:
    #                 pipeline = json.load(f)
    #                 try:
    #                     if 'pipeline_rank' in pipeline:
    #                         ranked_list.append((pipeline['pipeline_rank'], filename))
    #                     else:
    #                         # Move pipelines without scores to pipelines_searched directory
    #                         self._logger.info(f'Pipeline does not have score. id={pipeline["id"]}')
    #                         shutil.move(filepath, self.config.pipelines_searched_dir)
    #                 except:
    #                     self._logger.warning("Broken or unfinished pipeline: " + str(filepath))
    #     if not ranked_list:
    #         self._logger.warning('No ranked pipelines found.')
    #         return

    #     # Copy top 20 pipelines to pipelines_ranked directory
    #     sorted(ranked_list, key=operator.itemgetter(0))
    #     for _, filename in ranked_list[:20]:
    #         shutil.copy(os.path.join(self.config.pipelines_scored_dir, filename), self.config.pipelines_ranked_dir)

    # def _process_pipeline_submission_old2(self) -> None:
    #     output_dir = os.path.dirname(self.output_pipelines_dir)
    #     print("[PROSKA]:", output_dir)
    #     pipelines_root: str = os.path.join(output_dir, 'pipelines')
    #     executables_root: str = os.path.join(output_dir, 'executables')
    #     supporting_root: str = os.path.join(output_dir, 'supporting_files')
    #     # os.path.join(os.path.dirname(executables_root), 'pipelines')

    #     # Read all the json files in the pipelines
    #     piplines_name_list = os.listdir(pipelines_root)
    #     if len(piplines_name_list) < 20:
    #         for name in piplines_name_list:
    #             try:
    #                 with open(os.path.join(pipelines_root, name)) as f:
    #                     rank = json.load(f)['pipeline_rank']
    #             except:
    #                 os.remove(os.path.join(pipelines_root, name))
    #         return

    #     pipelines_df = pd.DataFrame(0.0, index=piplines_name_list, columns=["rank"])
    #     for name in piplines_name_list:
    #         try:
    #             with open(os.path.join(pipelines_root, name)) as f:
    #                 rank = json.load(f)['pipeline_rank']
    #             pipelines_df.at[name, 'rank'] = rank
    #         except:
    #             os.remove(os.path.join(pipelines_root, name))


    #     # sort them based on their rank field
    #     pipelines_df.sort_values(by='rank', ascending=True, inplace=True)

    #     # make sure that "pipeline_considered" directory exists
    #     considered_root = os.path.join(os.path.dirname(pipelines_root), 'pipelines_considered')
    #     try:
    #         os.mkdir(considered_root)
    #     except FileExistsError:
    #         pass

    #     # pick the top 20 and move the rest to "pipeline_considered" directory
    #     for name in pipelines_df.index[20:]:
    #         os.rename(src=os.path.join(pipelines_root, name),
    #                   dst=os.path.join(considered_root, name))

    #     # delete the exec and supporting files related the moved pipelines
    #     for name in pipelines_df.index[20:]:
    #         pipeName = name.split('.')[0]
    #         try:
    #             os.remove(os.path.join(executables_root, pipeName + '.json'))
    #         except FileNotFoundError:
    #             traceback.print_exc()
    #             pass

    #         try:
    #             shutil.rmtree(os.path.join(supporting_root, pipeName))
    #         except FileNotFoundError:
    #             traceback.print_exc()
    #             pass

    def _run_SerialBaseSearch(self, report_ensemble, *, one_pipeline_only=False):
        self._search_method.initialize_problem(
            template_list=self.template_list,
            performance_metrics=self.config.problem['problem']['performance_metrics'],
            problem=self.config.problem,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            ensemble_tuning_dataset=self.ensemble_dataset,
            output_directory=self.config.output_dir,
            start_time=self.config.start_time,
            timeout_sec=self.config.timeout_search,
            extra_primitive=self.extra_primitive,
        )
        # report = self._search_method.search(num_iter=50)
        report = self._search_method.search(num_iter=self.config.serial_search_iterations, one_pipeline_only=one_pipeline_only)
        if report_ensemble:
            report_ensemble['report'] = report
        self._log_search_results(report=report)

    def _run_ParallelBaseSearch(self, report_ensemble):
        self._search_method.initialize_problem(
            template_list=self.template_list,
            performance_metrics=self.config.problem['problem']['performance_metrics'],
            problem=self.config.problem,
            # updated v2019.11: now do not send these datasets, but will let subprocess to load instead
            # to reduce the size of each pickled subprocess and prevent multiprocess queue out ot space
            test_dataset1=None,#self.test_dataset1,
            train_dataset1=None,#self.train_dataset1,
            test_dataset2=None,#self.test_dataset2,
            train_dataset2=None,#self.train_dataset2,
            all_dataset=None,#self.all_dataset,
            ensemble_tuning_dataset=None,#self.ensemble_dataset,
            output_directory=self.config.output_dir,
            start_time=self.config.start_time,
            timeout_sec=self.config.timeout_search,
            extra_primitive=self.extra_primitive,
        )

        report = self._search_method.search(num_iter=1000)
        if report_ensemble:
            report_ensemble['report'] = report
        self._log_search_results(report=report)

        self._search_method.job_manager.reset()

    # def _run_RandomDimSearch(self, report_ensemble):
    #     # !! Need to updated
    #     self._search_method = RandomDimensionalSearch(
    #         template_list=self.template,
    #         performance_metrics=self.config.problem['problem']['performance_metrics'],
    #         problem=self.config.problem,
    #         test_dataset1=self.test_dataset1,
    #         train_dataset1=self.train_dataset1,
    #         test_dataset2=self.test_dataset2,
    #         train_dataset2=self.train_dataset2,
    #         all_dataset=self.all_dataset,
    #         ensemble_tuning_dataset=self.ensemble_dataset,
    #         output_directory=self.config.output_dir,
    #         log_dir=self.config.log_dir,
    #         num_proc=self.config.cpu,
    #         timeout=self.config.timeout_search,
    #         extra_primitive=self.extra_primitive,
    #     )
    #     report = self._search_method.search(num_iter=10)
    #     if report_ensemble:
    #         report_ensemble['report'] = report
    #     self._log_search_results(report=report)

    #     self._search_method.job_manager.reset()

    # def _run_BanditDimSearch(self, report_ensemble):
    #     # !! Need to updated
    #     self._search_method = BanditDimensionalSearch(
    #         template_list=self.template,
    #         performance_metrics=self.config.problem['problem']['performance_metrics'],
    #         problem=self.config.problem,
    #         test_dataset1=self.test_dataset1,
    #         train_dataset1=self.train_dataset1,
    #         test_dataset2=self.test_dataset2,
    #         train_dataset2=self.train_dataset2,
    #         all_dataset=self.all_dataset,
    #         ensemble_tuning_dataset = self.ensemble_dataset,
    #         output_directory=self.config.output_dir,
    #         log_dir=self.config.log_dir,
    #         num_proc=self.config.cpu,
    #         start_time=self.config.start_time,
    #         timeout=self.config.timeout_search,
    #         extra_primitive=self.extra_primitive,
    #     )
    #     report = self._search_method.search(num_iter=5)
    #     if report_ensemble:
    #         report_ensemble['report'] = report
    #     self._log_search_results(report=report)

    #     self._search_method.job_manager.reset()

    # def _run_MultiBanditSearch(self, report_ensemble):
    #     # !! Need to updated
    #     self._search_method = MultiBanditSearch(
    #         template_list=self.template,
    #         performance_metrics=self.config.problem['problem']['performance_metrics'],
    #         problem=self.config.problem,
    #         test_dataset1=self.test_dataset1,
    #         train_dataset1=self.train_dataset1,
    #         test_dataset2=self.test_dataset2,
    #         train_dataset2=self.train_dataset2,
    #         all_dataset=self.all_dataset,
    #         ensemble_tuning_dataset = self.ensemble_dataset,
    #         output_directory=self.config.output_dir,
    #         log_dir=self.config.log_dir,
    #         num_proc=self.config.cpu,
    #         start_time=self.config.start_time,
    #         timeout=self.config.timeout_search,
    #         extra_primitive=self.extra_primitive,
    #     )
    #     report = self._search_method.search(num_iter=30)
    #     if report_ensemble:
    #         report_ensemble['report'] = report
    #     self._log_search_results(report=report)

    #     self._search_method.job_manager.reset()

    """
        **********************************************************************
        Public method (in alphabet)
        # 1 . add_d3m_index_and_prediction_class_name
        2 . generate_configuration_space
        3 . initialize_from_config_for_evaluation
        4 . initialize_from_config_train_test
        5 . load_pipe_runtime
        6 . load_templates
        7 . remove_empty_targets
        8 . split_dataset
        9 . test
        10. test_fitted_pipeline
        11. train
        12. write_training_results
        13. shutdown

        Used by TA3
        1. get_candidates
        2. get_problem
        3. load_fitted_pipeline
        4. export_solution
        **********************************************************************
    """
    def initialize(self, config: DsboxConfig):
        '''
        This method should be called as soon as possible. Need to spawn all processes before grpc connection.
        '''
        self.config = config

        # Set runtime environment info before process forks
        if self.config.static_dir:
            FittedPipeline.runtime_setting = self.config.get_runtime_setting()

        use_multiprocessing = True

        # 2019.7.19: added here to let system always run with serial mode for some speical dataset
        try:
            try:
                import networkx
                from d3m import container
                loader = D3MDatasetLoader()
                json_file = os.path.abspath(self.config.dataset_schema_files[0])
                all_dataset_uri = 'file://{}'.format(json_file)
                inputs = loader.load(dataset_uri=all_dataset_uri)
                # inputs = Dataset object
                max_accept_graph_size_for_parallel = 4000
                for resource_id, resource in inputs.items():
                    if isinstance(resource, networkx.classes.graph.Graph):
                        edgelist = networkx.to_pandas_edgelist(resource)
                    if isinstance(resource, container.DataFrame) and inputs.metadata.has_semantic_type((resource_id,), 'https://metadata.datadrivendiscovery.org/types/EdgeList'):
                        edgelist = resource #self._update_edge_list(outputs, resource_id)
                graph_size = edgelist.shape[0]
            except:
                graph_size = None
            if graph_size and graph_size > max_accept_graph_size_for_parallel:
                self._logger.warning("Change to serial mode for the graph problem with size larger than " + str(max_accept_graph_size_for_parallel))
                self.config.search_method = "serial"
            # kyao 2019-7-24: Try parallel
            # if "LL0_acled" in self.config.problem['id'] or "LL1_VTXC_1343_cora" in self.config.problem['id']:
            #     self._logger.warning("Change to serial mode for the speical problem id: " + str(self.config.problem['id']))
            #     self.config.search_method = "serial"

        except:
            pass
        # END change for 2019.7.19

        if self.config.search_method == 'serial':
            self._search_method = TemplateSpaceBaseSearch()
            use_multiprocessing = False
        elif self.config.search_method == 'parallel':
            self._search_method = TemplateSpaceParallelBaseSearch(num_proc=self.config.cpu)
        # elif self.config.search_method == 'bandit':
        #     self._search_method = BanditDimensionalSearch(num_proc=self.config.cpu)
        else:
            self._search_method = TemplateSpaceParallelBaseSearch(num_proc=self.config.cpu)

        if self.do_ensemble_tune:
            # creat a special dictionary that can collect the results in each processes
            if use_multiprocessing:
                from multiprocessing import Manager
                m = Manager()
                self.report_ensemble = m.dict()
            self.ensemble_voting_candidate_choose_method = 'lastStep'
            # self.ensemble_voting_candidate_choose_method = 'resultSimilarity'

    @staticmethod
    def find_possible_candidate(supplied_data) -> typing.List[int]:
        """
        function used to find corresponding column numbers that we need for running wikifier
        """
        res_id, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=supplied_data, resource_id=None)
        all_columns = list(range(supplied_dataframe.shape[1]))
        target_columns = copy.deepcopy(all_columns)

        need_column_type = config_datamart.need_wikifier_column_type_list

        for each in all_columns:
            each_column_semantic_type = supplied_data.metadata.query((res_id, ALL_ELEMENTS, each))['semantic_types']
            # if the column type inside here found, this coumn should be wikified
            if set(each_column_semantic_type).intersection(need_column_type):
                continue
            else:
                target_columns.remove(each)

        return target_columns

    def run_wikifier(self, augment_res) -> str:
        """
            run wikifier before sending dataset to datamart, 
            then use similiarity of Q nodes vectors to determine which columns may be correct and useful
            return a jsonfied string that can be sent as keyword to let isi datamart system to 
            process with specified wikifier columns
        """
        # try to find target columns which should do wikifier
        target_columns = self.find_possible_candidate(augment_res)

        # get smaller dataset by random
        _, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=augment_res, resource_id=None)
        size_of_df = len(supplied_dataframe)
        if size_of_df > self.wikifier_max_len:
            size_of_sample = int(size_of_df * self.wikifier_selection_rate)
        elif size_of_df > self.wikifier_default_size:
            size_of_sample = self.wikifier_default_size
        else:
            size_of_sample = size_of_df
        random.seed(41)
        idx = random.sample(range(size_of_df), size_of_sample)

        # get qnode columns and metadata for wikifier
        meta_for_wikifier, sim_vector = dict(), dict()
        q_nodes_found_amount_in_sample_part = dict()

        for i in target_columns:
            sample_df = supplied_dataframe.iloc[idx, i].drop_duplicates(keep='first', inplace=False).to_frame()
            self._logger.info("Current column is " + str(sample_df.columns.tolist()))
            self._logger.debug("Start running wikifier...")
            output_df = wikifier.produce(sample_df, use_cache=False)
            self._logger.debug("Wikifier running finished.")

            if len(output_df.columns) > 1:
                # save specific p/q node in cache files
                res = MetadataCache.get_specific_p_nodes(sample_df)
                if res:
                    meta_for_wikifier.update(res)
                    MetadataCache.delete_specific_p_nodes_file(sample_df)
                # do vector augment and calculate cosine similarity
                qnode_name = output_df.columns.tolist()[1]
                q_nodes_found_amount_in_sample_part[qnode_name] = len(output_df[qnode_name].dropna())
                qnodes = output_df[qnode_name]
                sim_vector[qnode_name] = []
                df_vectors = DownloadManager.fetch_fb_embeddings(qnodes, qnode_name)
                for j in range(1, len(df_vectors.columns)):
                    # 201 columns: key, vector1, vector2 ...
                    sim_vector[qnode_name].append(df_vectors.iloc[:, j].mean())
            else:
                self._logger.info("This column do not has wikidata.")

        from sklearn.metrics.pairwise import cosine_similarity
        x = list(sim_vector.values())

        if len(x) != 0:
            matrix = cosine_similarity(x)
            df_sim = pd.DataFrame(data=matrix, columns=sim_vector.keys())

            # remove similar column
            # COMMENT: may remove right column when wrong columns are similar to each other.
            remove_set = set()
            col_name = df_sim.columns.tolist()

            for i, name in enumerate(col_name):
                if name not in remove_set:
                    candidate_column_need_drop = df_sim[name][(df_sim[name] > 0.9) | (df_sim[name] < 0.4)].index.tolist()
                    temp_q_nodes_amount_dict = dict()
                    for each_column in candidate_column_need_drop:
                        temp_q_nodes_amount_dict[col_name[each_column]] = q_nodes_found_amount_in_sample_part[col_name[each_column]]
                    temp_q_nodes_amount_dict.pop(max(temp_q_nodes_amount_dict.items(), key=operator.itemgetter(1))[0])
                    for each_key in temp_q_nodes_amount_dict.keys():
                        remove_set.add(each_key[:-9])
            for name in remove_set:
                if name in meta_for_wikifier.keys():
                    del meta_for_wikifier[name]
        
        meta_to_str = json.dumps({config_datamart.wikifier_column_mark: meta_for_wikifier})
        self._logger.info("Following columns should be wikified as:")
        self._logger.info(str(meta_to_str))
        return meta_to_str


    def do_data_augmentation_rest_api(self, input_all_dataset: Dataset) -> Dataset:
        """
        function that do augmentation from rest api(server) side
        """
        # with open("/Users/minazuki/Desktop/aug_40.pkl","rb") as f:
            # self.all_dataset = pickle.load(f)
        # return None

        augment_times = 0
        system_url = os.environ.get('DATAMART_URL_ISI')
        datamart_unit = rest.RESTDatamart(connection_url=system_url)
        augment_res = copy.copy(self.all_dataset)
        candidate_aug_res = []
        # meta_to_str = self.run_wikifier(augment_res)
        meta_to_str = ""

        try:
            keywords_from_data = self.config.problem["data_augmentation"][0]["keywords"]
        except:
            keywords_from_data = ["woreda", "ethiopia", "flood", "cases" ,"fever"]

        query_search = datamart.DatamartQuery(keywords=keywords_from_data + [meta_to_str], variables=None)

        search_unit = datamart_unit.search_with_data(query=query_search, 
                                                     supplied_data=self.all_dataset, 
                                                     run_wikifier=True,
                                                     consider_wikifier_columns_only=True, 
                                                     # if augment with time is set to true, consider time will be useless
                                                     augment_with_time=True,
                                                     # consider_time=False,
                                                     )
        all_results1 = search_unit.get_next_page()

        if all_results1 is None:
            self._logger.warning("No search result returned!")
            return self.all_dataset

        rest.pretty_print_search_results(all_results1)

        from common_primitives.datamart_augment import Hyperparams as hyper_augment, DataMartAugmentPrimitive
        hyper_augment_default = hyper_augment.defaults()
        hyper_augment_default = hyper_augment_default.replace({"system_identifier":"ISI"})

        def augment_test_worker(augment_num, res_dict, search_res, supplied_dataset):
            def pp(augment_res):
                return augment_primitive.produce(inputs=augment_res).value
            self._logger.info("Starting testing No.{} augment.".format(augment_num))
            hyper_temp = hyper_augment_default.replace({"search_result":search_res.serialize()})
            augment_primitive = DataMartAugmentPrimitive(hyperparams=hyper_temp)
            prev_augment_res = copy.copy(supplied_dataset)
            try:
                augment_res = timeout_call(3000, pp, [prev_augment_res])
                if type(augment_res) is str or augment_res is None:
                    self._logger.info("Agument No.{} failed with error {}".format(augment_num, str(augment_res)))
                    res_dict[augment_num] = False
                else:
                    _, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=augment_res, resource_id=None)
                    if supplied_dataframe.shape == original_shape:
                        res_dict[augment_num] = False
                        self._logger.info("Agument No.{} do not add any extra columns! Will ignore.".format(augment_num))
                    else:
                        self._logger.debug("Augmented dataset's shape is {}".format(str(supplied_dataframe.shape)))
                        res_dict[str(augment_num) + "_res"] = supplied_dataframe
                        res_dict[augment_num] = True
                        self._logger.info("Agument No.{} success".format(augment_num))
            except:
                self._logger.info("Agument No.{} failed with error".format(augment_num))
                res_dict[augment_num] = False
            return augment_res

        search_result_list = all_results1
        # augment_res_list = []
        jobs = []
        manager = multiprocessing.Manager()
        augment_dict = manager.dict()
        _, original_df = d3m_utils.get_tabular_resource(dataset=self.all_dataset, resource_id=None)
        original_shape = original_df.shape
        self._logger.debug("Original dataset's shape is {}".format(original_shape))

        augment_res = copy.copy(self.all_dataset)

        """
        augment one by one way
        for i, search_res in enumerate(search_result_list):
            p = multiprocessing.Process(target=augment_test_worker, args=(i, augment_dict, search_res, self.all_dataset))
            jobs.append(p)
            p.start()
        """

        for i, search_res in enumerate(search_result_list):
            try:
                augment_res = augment_test_worker(i, augment_dict, search_res, augment_res)
                candidate_aug_res.append(search_res)
            except:
                pass

        keywords_from_data = ["woreda", "ethiopia", "fuel", "unpaid", "unemployment"]

        query_search = datamart.DatamartQuery(keywords=keywords_from_data + [meta_to_str], variables=None)
        search_unit = datamart_unit.search_with_data(query=query_search, 
                                                     supplied_data=self.all_dataset, 
                                                     run_wikifier=True,
                                                     consider_wikifier_columns_only=True, 
                                                     # if augment with time is set to true, consider time will be useless
                                                     # augment_with_time=True,
                                                     consider_time=False,
                                                     )
        all_results2 = search_unit.get_next_page()
        
        rest.pretty_print_search_results(all_results2)

        augmented_id = set()
        for i, each_search_result in enumerate(all_results1):
            each_search_res_json = each_search_result.get_json_metadata()
            summary = each_search_res_json['summary']
            augmented_id.add(summary['Datamart ID'])

        for i, each_search_result in enumerate(all_results2):
            each_search_res_json = each_search_result.get_json_metadata()
            summary = each_search_res_json['summary']
            if summary['Datamart ID'] not in augmented_id:
                try:
                    augment_res = augment_test_worker(i, augment_dict, each_search_result, augment_res)
                    candidate_aug_res.append(each_search_result)
                except:
                    pass
            else:
                self._logger.info("No.{} augmented, will not augment again.".format(str(i)))

        # self.all_dataset = augment_res

        return candidate_aug_res

        """
        # !!! v2019.12.18 must need to change back later here!!!
        not_all_finished = True
        while not_all_finished:
            # check status each 10s
            time.sleep(10)
            not_all_finished = False
            for each_job in jobs:
                each_job.join(timeout=0)
                if each_job.is_alive():
                    not_all_finished = True
                    self._logger.info("Not all testing augment finished!")
                    break

        can_augment_result_number = []
        for key, val in augment_dict.items():
            if val is True:
                can_augment_result_number.append(key)
        can_augment_result_number.sort()
        filterd_results = []
        for each in can_augment_result_number:
            filterd_results.append(all_results1[each])

        rest.pretty_print_search_results(filterd_results)

        return filterd_results
        """


    def do_data_augmentation(self, input_all_dataset: Dataset) -> Dataset:
        """
            use datamart primitives to do data augmentation on given dataset
            return the augmented dataset (if success), otherwise return the original input
        """
        try:
            # initialize
            from datamart_isi import entries
            isi_datamart_url = "http://dsbox02.isi.edu:9001/blazegraph/namespace/datamart3/sparql"
            datamart_unit = entries.Datamart(connection_url=isi_datamart_url)
            from common_primitives.datamart_augment import Hyperparams as hyper_augment, DataMartAugmentPrimitive
            hyper_augment_default = hyper_augment.defaults()
            hyper_augment_default = hyper_augment_default.replace({"system_identifier":"ISI"})

            # run wikifier first
            augment_times = 0

            if self.all_dataset.metadata.query(())['id'].startswith("DA_medical_malpractice"):
                # # this special change only for running for DA_medical dataset, so that we can also use this column as a join candidate
                # # also, due to the reason that both supplied data and searched results are very large, skip wikidata part
                augment_res = self.all_dataset
                # meta =     {
                #      "name": "SEQNO",
                #      "structural_type": str,
                #      "semantic_types": [
                #       "http://schema.org/Text",
                #       "http://schema.org/DateTime",
                #       "https://metadata.datadrivendiscovery.org/types/UniqueKey"
                #      ],
                #      "description": "Record Number. SEQNO is a unique number assigned to each record. The assigned numbers are not necessarily continuous or sequential."
                #     }
                # augment_res.metadata = augment_res.metadata.update(selector=('learningData', ALL_ELEMENTS, 1), metadata = meta)
                search_unit = datamart_unit.search_with_data(query=None, supplied_data=augment_res, need_wikidata=False)

            elif self.all_dataset.metadata.query(())['id'].startswith("DA_ny_taxi_demand"):
                augment_res = self.all_dataset
                search_unit = datamart_unit.search_with_data(query=None, supplied_data=augment_res, need_wikidata=False)

            else:
                # in general condition, run wikifier first
                search_result_wikifier = entries.DatamartSearchResult(search_result={}, supplied_data=None, query_json={}, search_type="wikifier")
                hyper_temp = hyper_augment_default.replace({"search_result":search_result_wikifier.serialize()})
                augment_primitive = DataMartAugmentPrimitive(hyperparams=hyper_temp)
                augment_res = augment_primitive.produce(inputs = self.all_dataset).value
                # this part's code is only used for saving the pipeline afterwards in TA2 system
                self.extra_primitive.add("augment" + str(augment_times))
                self.dump_primitive(augment_primitive, "augment" + str(augment_times))
                augment_times += 1
                search_unit = datamart_unit.search_with_data(query=None, supplied_data=augment_res)

            # run search, it will return wikidata search results first (if found) and then the general search results with highest score first

            all_results1 = search_unit.get_next_page()

            for each_search in all_results1:
                if each_search.search_type == "wikidata" and len(each_search.search_result["p_nodes_needed"]) > 0:
                    hyper_temp = hyper_augment_default.replace({"search_result":each_search.serialize()})
                    augment_primitive = DataMartAugmentPrimitive(hyperparams=hyper_temp)
                    augment_res = augment_primitive.produce(inputs = augment_res).value
                    # this part's code is only used for saving the pipeline afterwards in TA2 system
                    self.extra_primitive.add("augment" + str(augment_times))
                    self.dump_primitive(augment_primitive, "augment" + str(augment_times))
                    augment_times += 1

            # you can search another time if you want
            # all_results2 = datamart_unit.search_with_data(query=None, supplied_data=augment_res).get_next_page()
            all_results1.sort(key=lambda x: x.score(), reverse=True)

            for each_search in all_results1:
                if each_search.search_type == "general":
                    # now only augment 1 times on gneral search results
                    hyper_temp = hyper_augment_default.replace({"search_result":each_search.serialize()})
                    augment_primitive = DataMartAugmentPrimitive(hyperparams=hyper_temp)
                    augment_res = augment_primitive.produce(inputs = augment_res).value
                    self.extra_primitive.add("augment" + str(augment_times))
                    self.dump_primitive(augment_primitive, "augment" + str(augment_times))
                    augment_times += 1
                    break

            # # return the augmented dataset
            original_shape = self.all_dataset[self.problem_info["res_id"]].shape
            _, augment_res_df = d3m_utils.get_tabular_resource(dataset=augment_res, resource_id=None)
            augmented_shape = augment_res_df.shape
            self._logger.info("The original dataset shape is (" + str(original_shape[0]) + ", " + str(original_shape[1]) + ")")
            self._logger.info("The augmented dataset shape is (" + str(augmented_shape[0]) + ", " + str(augmented_shape[1]) + ")")

            return augment_res

        except:
            self._logger.error("Agument Failed!")
            traceback.print_exc()
            return self.all_dataset

    # No longer needed
    # def add_d3m_index_and_prediction_class_name(self, prediction, from_dataset = None):
    #     """
    #         The function to process the prediction results
    #         1. If no prediction column name founnd, add the prediction column name
    #         2. Add the d3m index into the output predictions
    #     """
    #     # setup an initial condition
    #     if not from_dataset:
    #         from_dataset = self.all_dataset

    #     prediction_class_name = []
    #     try:
    #         with open(self.config.dataset_schema_files[0], 'r') as dataset_description_file:
    #             dataset_description = json.load(dataset_description_file)
    #             for each_resource in dataset_description["dataResources"]:
    #                 if "columns" in each_resource:
    #                     for each_column in each_resource["columns"]:
    #                         if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
    #                             prediction_class_name.append(each_column["colName"])
    #     except:
    #         self._logger.error(
    #             "[Warning] Can't find the prediction class name, will use default name "
    #             "'prediction'.")
    #         prediction_class_name.append("prediction")

    #     # if the prediction results do not have d3m_index column
    #     if 'd3mIndex' not in prediction.columns:
    #         d3m_index = get_target_columns(from_dataset)["d3mIndex"]
    #         d3m_index = d3m_index.reset_index().drop(columns=['index'])
    #         # prediction.drop("confidence", axis=1, inplace=True, errors = "ignore")#some
    #         # prediction has "confidence"
    #         prediction_col_name = ['d3mIndex']
    #         for each in prediction.columns:
    #             prediction_col_name.append(each)
    #         prediction['d3mIndex'] = d3m_index
    #         prediction = prediction[prediction_col_name]
    #         prediction_col_name.remove('d3mIndex')
    #         for i in range(len(prediction_class_name)):
    #             prediction = prediction.rename(
    #                 columns={prediction_col_name[i]: prediction_class_name[i]})
    #     else:
    #         prediction_col_name = list(prediction.columns)
    #         prediction_col_name.remove('d3mIndex')
    #         for i in range(len(prediction_class_name)):
    #             prediction = prediction.rename(
    #                 columns={prediction_col_name[i]: prediction_class_name[i]})
    #     return prediction

    def auto_regress_convert_and_add_metadata(self, dataset: Dataset):
        """
        Muxin said it is useless, just keep it for now
        """
        return dataset
        # """
        # Add metadata to the dataset from problem_doc_metadata
        # If the dataset is timeseriesforecasting, do auto convert for timeseriesforecasting prob
        # Paramters
        # ---------
        # dataset
        #     Dataset
        # problem_doc_metadata:
        #     Metadata about the problemDoc
        # """
        # problem = self.config.problem_metadata.query(())
        # targets = problem["inputs"]["data"][0]["targets"]
        # for each_target in range(len(targets)):
        #     resID = targets[each_target]["resID"]
        #     colIndex = targets[each_target]["colIndex"]
        #     if problem["about"]["taskType"] == "timeSeriesForecasting" or problem["about"][
        #         "taskType"] == "regression":
        #         dataset[resID].iloc[:, colIndex] = pd.to_numeric(dataset[resID].iloc[:, colIndex],
        #                                                          downcast="float", errors="coerce")
        #         meta = dict(dataset.metadata.query((resID, ALL_ELEMENTS, colIndex)))
        #         meta["structural_type"] = float
        #         dataset.metadata = dataset.metadata.update((resID, ALL_ELEMENTS, colIndex), meta)

        # for data in self.config.problem_metadata.query(())['inputs']['data']:
        #     targets = data['targets']
        #     for target in targets:
        #         semantic_types = list(dataset.metadata.query(
        #             (target['resID'], ALL_ELEMENTS, target['colIndex'])).get(
        #             'semantic_types', []))

        #         if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
        #             semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
        #             dataset.metadata = dataset.metadata.update(
        #                 (target['resID'], ALL_ELEMENTS, target['colIndex']),
        #                 {'semantic_types': semantic_types})

        #         if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
        #             semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        #             dataset.metadata = dataset.metadata.update(
        #                 (target['resID'], ALL_ELEMENTS, target['colIndex']),
        #                 {'semantic_types': semantic_types})
        #     return dataset

    def dump_primitive(self, target_primitive, save_file_name) -> bool:
        """
            Function used to dump a (usually it should be fitted) primitive into D#MLOCALDIR for further use
        """
        try:
            # pickle this fitted sampler for furture use in pipelines
            name = self.all_dataset.metadata.query(())['id']
            sampler_pickle_file_loc = os.path.join(self.config.dsbox_scratch_dir, name+save_file_name+".pkl")
            with open(sampler_pickle_file_loc, "wb") as f:
                pickle.dump(target_primitive, f)

            hyperparams_now = target_primitive.hyperparams.values_to_json_structure()
            sampler_hyperparams_file_loc = os.path.join(self.config.dsbox_scratch_dir, name+save_file_name+".json")
            with open(sampler_hyperparams_file_loc, "w") as f:
                json.dump(hyperparams_now, f)
            return True
        except:
            return False

    def ensemble_tuning(self, ensemble_tuning_report) -> None:
        """
        Function to do ensemble tuning

        :param ensemble_tuning_report: the report including the predictions for ensemble part's dataset,
                                       pipeline structure and metrics score of test datasets

        :return: None
        """
        if not self.ensemble_dataset:
            self._logger.error("No ensemble tuning dataset found!")

        elif not ensemble_tuning_report or 'report' not in ensemble_tuning_report:
            self._logger.error("No ensemble tuning inputs found!")

        else:
            try:
                pp = EnsembleTuningPipeline(pipeline_files_dir=self.config.output_dir,
                                            pids=None, candidate_choose_method=self.ensemble_voting_candidate_choose_method,
                                            report=ensemble_tuning_report, problem=self.problem,
                                            test_dataset=self.test_dataset1, train_dataset=self.train_dataset1)
                pp.generate_candidate_pids()
                pp.generate_ensemble_pipeline()
                pp.fit_and_produce()
                pp.save()
            except:
                self._logger.error("[ERROR] ensemble tuning pipeline failed.")
                traceback.print_exc()

    def horizontal_tuning(self, final_step_primitive) -> None:
        if not self.ensemble_dataset:
            self._logger.error("No ensemble tuning dataset found")
        else:
            try:
                qq = HorizontalTuningPipeline(pipeline_files_dir=self.config.output_dir,
                                              pids=None, problem=self.problem, test_dataset=self.test_dataset1,
                                              train_dataset=self.train_dataset1,
                                              final_step_primitive=final_step_primitive)
                qq.generate_candidate_pids()
                qq.generate_ensemble_pipeline()
                qq.fit_and_produce()
                qq.save()
            except:
                self._logger.error("[ERROR] horizontal tuning pipeline failed.")
                traceback.print_exc()

    def initialize_from_config_for_evaluation(self, config: DsboxConfig) -> None:
        """
            This function for running ta2_evaluation
        """
        self.config = config
        self._load_schema()
        self._log_init()

        # Dataset
        loader = D3MDatasetLoader()
        json_file = os.path.abspath(self.config.dataset_schema_files[0])
        all_dataset_uri = 'file://{}'.format(json_file)
        self.all_dataset = loader.load(dataset_uri=all_dataset_uri)
        self._check_and_set_dataset_metadata()
        # first apply denormalize on input dataset
        from common_primitives.denormalize import Hyperparams as hyper_denormalize, DenormalizePrimitive
        denormalize_hyperparams = hyper_denormalize.defaults()
        denormalize_primitive = DenormalizePrimitive(hyperparams = denormalize_hyperparams)
        self.all_dataset = denormalize_primitive.produce(inputs = self.all_dataset).value
        self.extra_primitive.add("denormalize")
        self.dump_primitive(denormalize_primitive, "denormalize")
        datamart_search_results = None
        # import pdb
        # pdb.set_trace()
        # if "data_augmentation" in self.config.problem.keys():
            # try:
        datamart_search_results = self.do_data_augmentation_rest_api(self.all_dataset)
            # except Exception as e:
                # datamart_search_results = None
                # self._logger.error("Running data augmentation failed!!!")
                # traceback.print_exc()
        # load templates
        self.load_templates(datamart_search_results)

    def initialize_from_config_train_test(self, config: DsboxConfig) -> None:
        """
            This function for running for ta2-search
        """
        self.config = config
        self._load_schema()
        self._log_init()

        # Dataset
        loader = D3MDatasetLoader()

        json_file = os.path.abspath(self.config.dataset_schema_files[0])
        all_dataset_uri = 'file://{}'.format(json_file)
        self.all_dataset = loader.load(dataset_uri=all_dataset_uri)

        # Templates
        self.load_templates()

    def initialize_from_ta3(self, config: DsboxConfig):
        """
        This function for TA3 GRPC connection, which is official D3M evaluation mechanism.
        """
        self.config = config
        self._load_schema(is_ta3=True)
        self._log_init()

        # Dataset
        loader = D3MDatasetLoader()

        json_file = self.config.dataset_schema_files[0]
        if json_file.startswith('file://'):
            self.all_dataset = loader.load(dataset_uri=json_file)
        else:
            json_file = os.path.abspath(json_file)
            self.all_dataset = loader.load(dataset_uri='file://{}'.format(json_file))
        self._check_and_set_dataset_metadata()

        # first apply denormalize on input dataset
        from common_primitives.denormalize import Hyperparams as hyper_denormalize, DenormalizePrimitive
        denormalize_hyperparams = hyper_denormalize.defaults()
        denormalize_primitive = DenormalizePrimitive(hyperparams=denormalize_hyperparams)
        self.all_dataset = denormalize_primitive.produce(inputs=self.all_dataset).value
        self.extra_primitive.add("denormalize")
        self.dump_primitive(denormalize_primitive, "denormalize")
        datamart_search_results = None
        if "data_augmentation" in self.config.problem.keys():
            datamart_search_results = self.do_data_augmentation_rest_api(self.all_dataset)
        # load templates
        self.load_templates(datamart_search_results)



    def load_pipe_runtime(self):
        dir = os.path.expanduser(self.config.output_dir + '/pipelines_fitted')
        read_pipeline_id = self.saved_pipeline_id
        if read_pipeline_id == "":
            self._logger.info(
                "[INFO] No specified pipeline ID found, will load the latest "
                "crated pipeline.")
            # if no pipeline ID given, load the newest created file in the
            # folder
            files = [os.path.join(dir, f) for f in os.listdir(dir)]
            files.sort(key=lambda f: os.stat(f).st_mtime)
            lastmodified = files[-1]
            read_pipeline_id = lastmodified.split('/')[-1].split('.')[0]

        pipeline_load = FittedPipeline.load(folder_loc=self.config.output_dir,
                                            fitted_pipeline_id=read_pipeline_id)
        return self.config.output_dir, pipeline_load, read_pipeline_id, pipeline_load.runtime

    def load_templates(self, datamart_search_results=None) -> None:

        self.template_list = self.template_library.get_templates(self.config.task_type,
                                                                 self.config.task_subtype,
                                                                 self.taskSourceType,
                                                                 self.specialized_problem)
        # find the maximum dataset split requirements
        for each_template in self.template_list:
            for each_step in each_template.template['steps']:
                if "runtime" in each_step and "test_validation" in each_step["runtime"]:
                    split_times = int(each_step["runtime"]["test_validation"])
                    if split_times > self.max_split_times:
                        self.max_split_times = split_times

        if datamart_search_results is not None and len(datamart_search_results) > 0:
            from dsbox.template.template_steps import TemplateSteps
            from dsbox.datapreprocessing.cleaner.splitter import SplitterHyperparameter
            splitter_hyperparam = SplitterHyperparameter.defaults()
            large_dataset_row_length = splitter_hyperparam['threshold_row_length']
            large_dataset_column_length = splitter_hyperparam['threshold_column_length']
            res_id, supplied_dataframe = d3m_utils.get_tabular_resource(dataset=self.all_dataset, resource_id=None)
            is_large_dataset = supplied_dataframe.shape[0] >= large_dataset_row_length or supplied_dataframe.shape[1] >= large_dataset_column_length
            if is_large_dataset:
                self._logger.info("Large dataset detected! Will skip wikidata related parts!")
            augment_steps = TemplateSteps.dsbox_augmentation_step(datamart_search_results, large_dataset=is_large_dataset, augment_algorithm="augment_all_in_one")
            self._logger.info("Totally " + str(len(augment_steps)) + " datamart search results will be considered!")

            for each_template in self.template_list:
                if "gradient" in each_template.template['name'] or "default" in each_template.template['name']:
                    # remove to dataframe step
                    if each_template.template['steps'][0]['name'] == 'to_dataframe_step':
                        each_template.template['steps'].pop(0)
                    each_template.template['steps'][0]['inputs'] = [augment_steps[-1]['name']]
                    each_template.template['steps'] = augment_steps + each_template.template['steps']
                    self._logger.info("Extra augmentation steps has been added for template " + each_template.template['name'])


    def remove_empty_targets(self, dataset: Dataset) -> Dataset:
        """
        will automatically remove empty targets in training
        """
        problem = self.config.problem

        # do not remove columns for cluster dataset!
        if TaskKeyword.CLUSTERING in problem['problem']['task_keywords']:
            return dataset

        resID, _ = d3m_utils.get_tabular_resource(dataset=dataset, resource_id=None)
        # targets = list(dataset.metadata.list_columns_with_semantic_types(
        #     ['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
        #     at=(resID,),
        # ))
        # colIndex = targets[0]

        # TODO: update to use D3M's method to accelerate the processing speed

        # droplist = []
        # for i, v in dataset[resID].iterrows():
        #     if v[colIndex] == "":
        #         droplist.append(i)
        # if droplist != []:
        #     self._logger.warning('!!!! THIS IS MOST LIKELY A SEMI-SUPERVISED LEARNING DATASET !!!!')
        #     dataset[resID] = dataset[resID].drop(dataset[resID].index[droplist])
        #     meta = dict(dataset.metadata.query((resID,)))
        #     dimension = dict(meta['dimension'])
        #     meta['dimension'] = dimension
        #     dimension['length'] = dataset[resID].shape[0]
        #     dataset.metadata = dataset.metadata.update((resID,), meta)

        return dataset

    def split_dataset(self, dataset, random_state=42, test_size=0.2, n_splits=1, need_test_dataset=True):
        """
            Split dataset into 2 parts for training and test
        """
        '''
        def _add_meta_data(dataset, res_id, input_part):
            dataset_with_new_meta = copy.copy(dataset)
            dataset_metadata = dict(dataset_with_new_meta.metadata.query(()))
            dataset_metadata['id'] = dataset_metadata['id'] + '_' + str(uuid.uuid4())
            dataset_with_new_meta.metadata = dataset_with_new_meta.metadata.update((),
                                                                                   dataset_metadata)

            dataset_with_new_meta[res_id] = input_part
            meta = dict(dataset_with_new_meta.metadata.query((res_id,)))
            dimension = dict(meta['dimension'])
            meta['dimension'] = dimension
            dimension['length'] = input_part.shape[0]
            # print(meta)
            dataset_with_new_meta.metadata = dataset_with_new_meta.metadata.update((res_id,), meta)
            # pprint(dict(dataset_with_new_meta.metadata.query((res_id,))))
            return dataset_with_new_meta
        '''
        task_type = self.problem_info["task_type"]  # ['problem']['task_type'].name  # 'classification' 'regression'
        if not isinstance(task_type, list): 
            task_type = [task_type]
        # res_id = self.problem_info["res_id"]
        # target_index = self.problem_info["target_index"]
        data_type = self.problem_info["data_type"]

        cannot_split = False

        for each in data_type:
            if each in self.data_type_cannot_split:
                cannot_split = True
                break

        # check second time if the program think we still can split
        if not cannot_split:
            task_type_check = task_type

            for each in task_type_check:
                if each not in self.task_type_can_split:
                    cannot_split = True
                    break

        # if the dataset type in the list that we should not split
        if cannot_split:
            train_return = []
            test_return = []
            for i in range(n_splits):
                # just return all dataset to train part
                train_return.append(dataset)
                test_return.append(None)

        # if the dataset type can be split
        else:
            self._logger.info("split start!")
            train_ratio = 1 - test_size
            if n_splits == 1:
                from common_primitives.train_score_split import TrainScoreDatasetSplitPrimitive, Hyperparams as hyper_train_split
                hyperparams_split = hyper_train_split.defaults()
                hyperparams_split = hyperparams_split.replace({"train_score_ratio": train_ratio, "shuffle": True})
                if 'CLASSIFICATION' in task_type:
                    hyperparams_split = hyperparams_split.replace({"stratified": True})
                else:  # if not task_type == "REGRESSION":
                    hyperparams_split = hyperparams_split.replace({"stratified": False})
                split_primitive = TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_split)

            else:
                from common_primitives.kfold_split import KFoldDatasetSplitPrimitive, Hyperparams as hyper_k_fold
                hyperparams_split = hyper_k_fold.defaults()
                hyperparams_split = hyperparams_split.replace({"number_of_folds":n_splits, "shuffle":True})
                if 'CLASSIFICATION' in task_type:
                    hyperparams_split = hyperparams_split.replace({"stratified":True})
                else:# if not task_type == "REGRESSION":
                    hyperparams_split = hyperparams_split.replace({"stratified":False})
                split_primitive = KFoldDatasetSplitPrimitive(hyperparams = hyperparams_split)

            try:
                split_primitive.set_training_data(dataset = dataset)
                split_primitive.fit()
                # TODO: is it correct here?
                query_dataset_list = list(range(n_splits))
                train_return = split_primitive.produce(inputs = query_dataset_list).value#['learningData']
                test_return = split_primitive.produce_score_data(inputs = query_dataset_list).value

            except Exception:
                # Do not split stratified shuffle fails
                train_return = []
                test_return = []
                self._logger.info('Not splitting dataset. Stratified shuffle failed')
                for i in range(n_splits):
                    train_return.append(dataset)
                    test_return.append(None)

            self._logger.info("split done!")


            '''
            # old method (achieved by ourselves) to generate splitted datasets

            if task_type == 'CLASSIFICATION':
                self._logger.info("split start!!!!!!")
                try:
                    # Use stratified sample to split the dataset
                    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                                 random_state=random_state)
                    sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])

                    for train_index, test_index in sss.split(dataset[res_id],
                                                             dataset[res_id].iloc[:, target_index]):
                        indf = dataset[res_id]
                        outdf_train = pd.DataFrame(columns=dataset[res_id].columns)

                        for each_index in train_index:
                            outdf_train = outdf_train.append(indf.loc[each_index],
                                                             ignore_index=True)

                        # reset to sequential
                        outdf_train = outdf_train.reset_index(drop=True)

                        outdf_train = d3m_DataFrame(outdf_train, generate_metadata=False)
                        train = _add_meta_data(dataset=dataset, res_id=res_id,
                                               input_part=outdf_train)
                        train_return.append(train)

                        # for special condition that only need get part of the dataset
                        if need_test_dataset:
                            outdf_test = pd.DataFrame(columns=dataset[res_id].columns)
                            for each_index in test_index:
                                outdf_test = outdf_test.append(indf.loc[each_index],
                                                               ignore_index=True)
                            # reset to sequential
                            outdf_test = outdf_test.reset_index(drop=True)
                            outdf_test = d3m_DataFrame(outdf_test, generate_metadata=False)
                            test = _add_meta_data(dataset=dataset, res_id=res_id,
                                                  input_part=outdf_test)
                            test_return.append(test)
                        else:
                            test_return.append(None)

                    self._logger.info("split done!!!!!!")
                except Exception:
                    # Do not split stratified shuffle fails
                    self._logger.info('Not splitting dataset. Stratified shuffle failed')
                    for i in range(n_splits):
                        train_return.append(dataset)
                        test_return.append(None)

            else:
                # Use random split
                if not task_type == "REGRESSION":
                    print('USING Random Split to split task type: {}'.format(task_type))
                ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                ss.get_n_splits(dataset[res_id])
                for train_index, test_index in ss.split(dataset[res_id]):
                    train = _add_meta_data(dataset=dataset, res_id=res_id,
                                           input_part=dataset[res_id].iloc[train_index, :].reset_index(drop=True))
                    train_return.append(train)
                    # for special condition that only need get part of the dataset
                    if need_test_dataset:
                        test = _add_meta_data(dataset=dataset, res_id=res_id,
                                              input_part=dataset[res_id].iloc[test_index, :].reset_index(drop=True))
                        test_return.append(test)
                    else:
                        test_return.append(None)
            '''
        return train_return, test_return

    def test(self) -> Status:
        """
            First read the fitted pipeline and then run trained pipeline on test data.
        """
        self._logger.info("[INFO] Start test function")
        outputs_loc, pipeline_load, read_pipeline_id, run_test = self.load_pipe_runtime()

        self._logger.info("[INFO] Pipeline load finished")

        self._logger.info("[INFO] testing data")

        run_test.produce(inputs=[self.all_dataset])

        # try:
        #     step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        # except:
        #     self._logger.error("Warning: searching the output step number failed! "
        #                        "Will use the last step's output of the pipeline.")
        #     # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            # step_number_output = len(run_test.produce_outputs) - 1

        # update: it seems now prediction on new runtime will only have last output
        # TODO: check whether it fit all dataset
        # step_number_output = 0
        # prediction = run_test.produce_outputs[step_number_output]
        # prediction = self.add_d3m_index_and_prediction_class_name(prediction)

        if 'outputs.0' not in run_test.produce_outputs.values:
            self._logger.error("Could not find 'outputs.' in pipeline outputs")
            return Status.ERROR

        prediction = run_test.produce_outputs.values['outputs.0']
        prediction_folder_loc = outputs_loc + "/predictions/" + read_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)

        self._logger.info("[INFO] Finished: prediction results saving finished")
        self._logger.info(
            "[INFO] The prediction results is stored at: {}".format(prediction_folder_loc))
        return Status.OK

    def test_fitted_pipeline(self, fitted_pipeline_id):
        print("[INFO] Start test function")
        d = os.path.expanduser(self.config.output_dir + '/pipelines')
        if fitted_pipeline_id == "":
            print(
                "[INFO] No specified pipeline ID found, will load the latest "
                "crated pipeline.")
            # if no pipeline ID given, load the newest created file in the
            # folder
            files = [os.path.join(d, f) for f in os.listdir(d)]
            files.sort(key=lambda f: os.stat(f).st_mtime)
            lastmodified = files[-1]
            fitted_pipeline_id = lastmodified.split('/')[-1].split('.')[0]

        pipeline_load = FittedPipeline.load(folder_loc=self.config.output_dir,
                                            fitted_pipeline_id=fitted_pipeline_id)
        run_test = pipeline_load.runtime
        print("[INFO] Pipeline load finished")

        print("[INFO] testing data:")
        # pprint(self.test_dataset.head())

        # pipeline_load.runtime.produce(inputs=[self.test_dataset])
        # runtime.add_target_columns_metadata(self.all_dataset, self.config.problem)
        run_test.produce(inputs=[self.all_dataset])

        # try:
        #     step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        # except:
        #     self._logger.error("Warning: searching the output step number failed! "
        #                        "Will use the last step's output of the pipeline.")
        #     # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
        #     step_number_output = len(run_test.produce_outputs) - 1
        step_number_output = 0
        # get the target column name
        prediction_class_name = []
        try:
            with open(self.config.dataset_schema_files[0], 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_resource in dataset_description["dataResources"]:
                    if "columns" in each_resource:
                        for each_column in each_resource["columns"]:
                            if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
                                prediction_class_name.append(each_column["colName"])
        except Exception:
            self._logger.error(
                "[Warning] Can't find the prediction class name, will use default name "
                "'prediction'.")
            prediction_class_name.append("prediction")

        prediction = run_test.produce_outputs[step_number_output]
        # Should not fix predictions with d3m_index column
        # # if the prediction results do not have d3m_index column
        # if 'd3mIndex' not in prediction.columns:
        #     d3m_index = get_target_columns(self.all_dataset, self.config.problem)["d3mIndex"]
        #     d3m_index = d3m_index.reset_index().drop(columns=['index'])
        #     # prediction.drop("confidence", axis=1, inplace=True, errors = "ignore")#some
        #     # prediction has "confidence"
        #     prediction_col_name = ['d3mIndex']
        #     for each in prediction.columns:
        #         prediction_col_name.append(each)
        #     prediction['d3mIndex'] = d3m_index
        #     prediction = prediction[prediction_col_name]
        #     prediction_col_name.remove('d3mIndex')
        #     for i in range(len(prediction_class_name)):
        #         prediction = prediction.rename(
        #             columns={prediction_col_name[i]: prediction_class_name[i]})
        prediction_folder_loc = self.config.output_dir + "/predictions/" + fitted_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)

        # prediction = self.add_d3m_index_and_prediction_class_name(prediction)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)
        self._logger.info("[INFO] Finished: prediction results saving finished")
        self._logger.info(
            "[INFO] The prediction results is stored at: {}".format(prediction_folder_loc))
        return Status.OK

    def write_training_results(self):
        # load trained pipelines
        if os.getpid() == self.main_pid:
            self._logger.warning("write_training_results")
            self._did_we_post_process = True
            self._process_pipeline_submission()

        return None

    def shutdown(self):
        """
        Gracefully shutdown.
        """
        self._search_method.shutdown()

    def train(self, *, one_pipeline_only=False) -> Status:
        """
        Generate and train pipelines.
        """
        logging.getLogger("d3m").setLevel(logging.ERROR)
        if not self.template_list:
            return Status.PROBLEM_NOT_IMPLEMENT

        self.generate_dataset_splits()

        # FIXME) come up with a better way to implement this part. The fork does not provide a way
        # FIXME) to catch the errors of the child process
        self.resource_monitor.start_recording_resource_usage()

        if self.config.search_method == 'serial':
            self._run_SerialBaseSearch(self.report_ensemble, one_pipeline_only=one_pipeline_only)
        else:
            self._run_ParallelBaseSearch(self.report_ensemble)

        # if not self.use_multiprocessing or self.config.search_method == 'serial':
        #     self._run_SerialBaseSearch(self.report_ensemble, one_pipeline_only=one_pipeline_only)
        # else:
        #     from multiprocessing import Process
        #     with mplog.open_queue() as log_queue:
        #         self._logger.info('Starting Search process')

        #         if self.config.search_method == 'parallel':
        #             proc = Process(target=mplog.logged_call,
        #                            args=(log_queue, self._run_ParallelBaseSearch, self.report_ensemble))
        #         elif self.config.search_method == 'bandit':
        #             proc = Process(target=mplog.logged_call,
        #                            args=(log_queue, self._run_BanditDimSearch,))
        #         else:
        #             proc = Process(target=mplog.logged_call,
        #                            args=(log_queue, self._run_ParallelBaseSearch, self.report_ensemble))

        #         proc.start()
        #         self._logger.info('Searching is finished')

        #         # wait until process is done
        #         proc.join()
        #         print(f"END OF FORK {proc.exitcode}")
        #         status = proc.exitcode
        #         print("[INFO] Search Status:")
        #         pprint.pprint(status)

        if self.do_ensemble_tune:
            self._logger.info("Normal searching finished, now starting ensemble tuning")
            self.ensemble_tuning(self.report_ensemble)

        if self.do_horizontal_tune:
            self._logger.info("Starting horizontal tuning")
            self.horizontal_tuning("d3m.primitives.sklearn_wrap.SKBernoulliNB")

        self.resource_monitor.stop_recording_resource_usage(self.config.output_dir)
        self.write_training_results()
        return Status.OK

    def generate_dataset_splits(self):

        self.all_dataset = self.remove_empty_targets(self.all_dataset)
        from dsbox.datapreprocessing.cleaner.splitter import Splitter, SplitterHyperparameter

        hyper_sampler = SplitterHyperparameter.defaults()
        # for test purpose here
        hyper_sampler = hyper_sampler.replace({"threshold_column_length":2000,"further_reduce_threshold_column_length":2000})
        sampler = Splitter(hyperparams = hyper_sampler)
        sampler.set_training_data(inputs = self.all_dataset)
        sampler.fit()
        train_split = sampler.produce(inputs = self.all_dataset)

        _, original_df = d3m_utils.get_tabular_resource(dataset=self.all_dataset, resource_id=None)
        _, split_df = d3m_utils.get_tabular_resource(dataset=train_split.value, resource_id=None)
        if original_df.shape != split_df.shape:
            self.extra_primitive.add("splitter")
            self.all_dataset = train_split.value
            # pickle this fitted sampler for furture use in pipelines
            self.dump_primitive(sampler,"splitter")

        '''
        # old method here

        # runtime.add_target_columns_metadata(self.all_dataset, self.config.problem)
        res_id = self.problem_info['res_id']
        # check the shape of the dataset
        main_res_shape = self.all_dataset[res_id].shape
        # if the column length is larger than the threshold, it may failed in the given time,
        # so we need to sample part of the dataset

        if main_res_shape[1] > self.threshold_column_length:
            self._logger.info(
                "The columns number of the input dataset is very large, now sampling part of them.")

            # first check the target column amount
            target_column_list = []
            all_column_length = \
                self.all_dataset.metadata.query((res_id, ALL_ELEMENTS))['dimension']['length']

            targets_from_problem = self.config.problem_metadata.query(())["inputs"]["data"][0][
                "targets"]
            for t in targets_from_problem:
                target_column_list.append(t["colIndex"])
            self._logger.info("Totally {} taget found.".format(len(target_column_list)))
            target_column_length = len(target_column_list)

            # check again on the length of the column to ensure
            if (main_res_shape[1] - target_column_length - 1) <= self.threshold_column_length:
                pass
            else:
                # TODO: current large dataset processing function is not fully finished!!!
                attribute_column_length = all_column_length - target_column_length - 1
                # skip the column 0 which is d3mIndex]
                is_all_numerical = True
                # check whether all inputs are categorical or not
                # for each_column in range(1, attribute_column_length + 1):
                #     each_metadata = self.all_dataset.metadata.query((res_id,ALL_ELEMENTS,
                # each_column))
                #     if 'http://schema.org/Float' not in each_metadata['semantic_types'] or
                # 'http://schema.org/Integer' not in each_metadata['semantic_types']:
                #         is_all_numerical = False
                #         break
                # two ways to do sampling (random projection or random choice)
                if is_all_numerical:
                    # TODO:
                    # add special template that use random projection directly
                    # add one special source type for the template special process such kind of
                    # dataset
                    self._logger.info(
                        "Special type of dataset: large column number with all categorical "
                        "columns.")
                    self._logger.info("Will reload the template with new task source type.")
                    self.taskSourceType.add("large_column_number")
                    # aadd new template specially for large column numbers at the first priority
                    new_template = self.template_library.get_templates(self.config.task_type,
                                                                       self.config.task_subtype,
                                                                       self.taskSourceType)
                    # find the maximum dataset split requirements
                    for each_template in new_template:
                        self.template.insert(0, each_template)
                        for each_step in each_template.template['steps']:
                            if "runtime" in each_step and "test_validation" in each_step["runtime"]:
                                split_times = int(each_step["runtime"]["test_validation"])
                                if split_times > self.max_split_times:
                                    self.max_split_times = split_times

                    # else:
                    # run sampling method to randomly throw some columns
                    all_attribute_columns_list = set(range(1, all_column_length))
                    for each in target_column_list:
                        all_attribute_columns_list.remove(each)

                    # generate new metadata
                    metadata_new = DataMetadata()
                    metadata_old = copy.copy(self.all_dataset.metadata)

                    # generate the remained column index randomly and sort it
                    remained_columns = random.sample(all_attribute_columns_list,
                                                     self.threshold_column_length)
                    remained_columns.sort()
                    remained_columns.insert(0, 0)  # add column 0 (index column)
                    remained_columns.extend(target_column_list)  # add target columns
                    # sample the dataset
                    self.all_dataset[res_id] = self.all_dataset[res_id].iloc[:, remained_columns]

                    new_column_meta = dict(self.all_dataset.metadata.query((res_id, ALL_ELEMENTS)))
                    new_column_meta['dimension'] = dict(new_column_meta['dimension'])
                    new_column_meta['dimension'][
                        'length'] = self.threshold_column_length + 1 + target_column_length
                    # update whole source description
                    metadata_new = metadata_new.update((), metadata_old.query(()))
                    metadata_new = metadata_new.update((res_id,), metadata_old.query((res_id,)))
                    metadata_new = metadata_new.update((res_id, ALL_ELEMENTS), new_column_meta)

                    # update the metadata on each column remained
                    metadata_new_target = {}
                    for new_column_count, each_remained_column in enumerate(remained_columns):
                        old_selector = (res_id, ALL_ELEMENTS, each_remained_column)
                        new_selector = (res_id, ALL_ELEMENTS, new_column_count)
                        metadata_new = metadata_new.update(new_selector,
                                                           metadata_old.query(old_selector))
                        # save the new target metadata
                        if new_column_count > self.threshold_column_length:
                            metadata_old.query(old_selector)['name']
                            metadata_new_target[
                                metadata_old.query(old_selector)['name']] = new_column_count
                    # update the new metadata to replace the old one
                    self.all_dataset.metadata = metadata_new
                    # update traget_index for spliting into train and test dataset
                    if type(self.problem_info["target_index"]) is list:
                        for i in range(len(self.problem_info["target_index"])):
                            self.problem_info["target_index"][
                                i] = self.threshold_column_length + i + 1
                    else:
                        self.problem_info[
                            "target_index"] = self.threshold_column_length + target_column_length

                    # update problem metadata
                    problem = dict(self.config.problem_metadata.query(()))
                    # data_meta = dict(problem["inputs"]["data"][0])
                    data_meta = []
                    for each_data in problem["inputs"]["data"]:
                        # update targets metadata for each target columns
                        target_meta = []
                        each_data = dict(each_data)
                        for each_target in each_data["targets"]:
                            target_meta_each = dict(each_target)
                            if target_meta_each['colName'] in metadata_new_target:
                                target_meta_each['colIndex'] = metadata_new_target[
                                    target_meta_each['colName']]
                            else:
                                self._logger.error("New target column for {} not found:".format(
                                    target_meta_each['colName']))
                            # target_meta_each['colIndex'] = self.threshold_column_length + (
                            # all_column_length - target_meta_each['colIndex'])
                            target_meta.append(frozendict.FrozenOrderedDict(target_meta_each))
                        # return the updated target_meta
                        each_data["targets"] = tuple(target_meta)
                        data_meta.append(each_data)
                    # return the updated data_meta
                    problem["inputs"] = dict(problem["inputs"])
                    problem["inputs"]["data"] = tuple(data_meta)

                    problem["inputs"] = frozendict.FrozenOrderedDict(problem["inputs"])
                    problem = frozendict.FrozenOrderedDict(problem)

                    # update problem doc metadata

                    # TODO: self.problem_doc_metadata moved to DsboxConfig
                    self.problem_doc_metadata = self.problem_doc_metadata.update((), problem)
                    # updating problem_doc_metadata finished

                    self._logger.info("Random sampling on columns Finished.")

        if main_res_shape[0] > self.threshold_index_length:
            self._logger.info(
                "The row number of the input dataset is very large, will send only part of them "
                "to search.")
            if main_res_shape[1] > 20:
                self.threshold_index_length = int(self.threshold_index_length * 0.3)
                self._logger.info(
                    "The column number is also very large, will reduce the sampling amount on row "
                    "number.")
            # too many indexs, we can run another split dataset
            index_removed_percent = 1 - float(self.threshold_index_length) / float(
                main_res_shape[0])
            # ignore the test part
            self.all_dataset, _ = self.split_dataset(dataset=self.all_dataset,
                                                     test_size=index_removed_percent,
                                                     need_test_dataset=False)
            self.all_dataset = self.all_dataset[0]
            self._logger.info("Random sampling on rows Finished.")
        '''

        # if we need to do ensemble tune, we split one extra time
        if self.do_ensemble_tune or self.do_horizontal_tune:
            self.train_dataset1, self.ensemble_dataset = self.split_dataset(dataset=self.all_dataset, test_size = 0.1)
            self.train_dataset1 = self.train_dataset1[0]
            self.ensemble_dataset = self.ensemble_dataset[0]
            self.train_dataset1, self.test_dataset1 = self.split_dataset(dataset=self.train_dataset1)

        else:
            # split the dataset first time
            self.train_dataset1, self.test_dataset1 = self.split_dataset(dataset=self.all_dataset, test_size = 0.1)
            if self._logger.getEffectiveLevel() <= 10:
                self._save_dataset(self.train_dataset1, pathlib.Path(self.config.dsbox_scratch_dir) / 'train_dataset1')
                self._save_dataset(self.test_dataset1, pathlib.Path(self.config.dsbox_scratch_dir) / 'test_dataset1')

        # here we only split one times, so no need to use list to include the dataset
        if len(self.train_dataset1) == 1:
            self.train_dataset1 = self.train_dataset1[0]
        else:
            self._logger.error("Some error happend with all_dataset split: "
                               "The length of splitted dataset is not 1 but %s",
                               len(self.train_dataset1))

        if len(self.test_dataset1) == 1:
            self.test_dataset1 = self.test_dataset1[0]
        else:
            self._logger.error("Split failed on all_dataset.")
            self.test_dataset1 = None

        # if necessary, we need to make a second split
        if self.max_split_times > 0:
            # make n times of different spliting results
            if self.max_split_times == 5:
                test_size = 0.2
            else:
                test_size = 0.1

            self.train_dataset2, self.test_dataset2 = self.split_dataset(
                dataset=self.train_dataset1, test_size=test_size, n_splits=self.max_split_times)
            if len(self.train_dataset2) < 1:
                self._logger.error(
                    "Some error happend with train_dataset1 split: The length of splitted dataset "
                    "is less than 1")
            if len(self.test_dataset2) < 1:
                self._logger.error("Split failed on train_dataset1.")
                self.test_dataset2 = None
        else:
            self.train_dataset2 = None
            self.test_dataset2 = None
        from dsbox.combinatorial_search.search_utils import save_pickled_dataset
        save_pickled_dataset(self.train_dataset1, "train_dataset1")
        save_pickled_dataset(self.train_dataset2, "train_dataset2")
        save_pickled_dataset(self.test_dataset1, "test_dataset1")
        save_pickled_dataset(self.test_dataset2, "test_dataset2")
        save_pickled_dataset(self.all_dataset, "all_dataset")
        save_pickled_dataset(self.ensemble_dataset, "ensemble_tuning_dataset")


    def _save_dataset(self, dataset_list: typing.List[Dataset], save_dir: pathlib.Path):
        if save_dir.exists():
            shutil.rmtree(save_dir)
        else:
            save_dir.mkdir()
        try:
            for i, dataset in enumerate(dataset_list):
                dataset_dir = save_dir / f'dataset_{i}'
                if dataset is None:
                    self._logger.warn(f'Data is none for {save_dir}')
                else:
                    dataset.save((dataset_dir / "datasetDoc.json").as_uri())
        except Exception:
            self._logger.debug("Failed to save dataset splits", exc_info=True)

    # Methods used by TA3

    def get_candidates(self) -> typing.Dict:
        return self._search_method.history.all_reports

    def get_problem(self) -> typing.Dict:
        return self.config.problem

    def load_fitted_pipeline(self, fitted_pipeline_id) -> FittedPipeline:
        fitted_pipeline_load = FittedPipeline.load(folder_loc=self.config.output_dir,
                                                   fitted_pipeline_id=fitted_pipeline_id)
        return fitted_pipeline_load

    def export_solution(self, fitted_pipeline_id) -> None:
        '''
        Copy pipeline to pipelines_ranked directory
        '''
        fitted_filepath = os.path.join(self.config.pipelines_fitted_dir, fitted_pipeline_id, fitted_pipeline_id + '.json')
        if not os.path.exists(fitted_filepath):
            self._logger.error(f'Fitted pipeline does not exists: {fitted_pipeline_id}')
            return

        with open(fitted_filepath) as f:
            fitted_structure = json.load(f)

        pipeline_id = fitted_structure['pipeline_id']
        filepath = os.path.join(self.config.pipelines_scored_dir, pipeline_id + '.json')

        if not os.path.exists(filepath):
            self._logger.error(f'Pipeline does not exists: {fitted_pipeline_id}')
            return

        if os.path.exists(os.path.join(self.config.pipelines_ranked_dir, pipeline_id + '.json')):
            self._logger.info(f'Pipeline solution already exported: {fitted_pipeline_id}')
            return

        shutil.copy(filepath, self.config.pipelines_ranked_dir)
