import enum
import json
import logging
import os
import random
import sys
import typing
import uuid
import json
import shutil

from math import sqrt, log
import traceback

from multiprocessing import Process

import numpy as np
import pandas as pd
import frozendict

import d3m
import dsbox.template.runtime as runtime

from d3m.metadata.problem import TaskType
from d3m.container.pandas import DataFrame as d3m_DataFrame
from d3m.container.dataset import Dataset
from d3m.container.dataset import D3MDatasetLoader
from d3m.exceptions import NotSupportedError
from d3m.exceptions import InvalidArgumentValueError
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.base import Metadata, DataMetadata
from d3m.metadata.problem import TaskSubtype
from d3m.metadata.problem import parse_problem_description

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.pipeline.utils import larger_is_better
from dsbox.schema.problem import optimization_type
from dsbox.schema.problem import OptimizationType
from dsbox.template.library import TemplateDescription
from dsbox.template.library import TemplateLibrary
from dsbox.template.library import SemanticTypeDict
from dsbox.template.configuration_space import ConfigurationSpace
from dsbox.template.configuration_space import SimpleConfigurationSpace
from dsbox.combinatorial_search.TemplateSpaceBaseSearch import TemplateSpaceBaseSearch
from dsbox.combinatorial_search.TemplateSpaceParallelBaseSearch import \
    TemplateSpaceParallelBaseSearch
from dsbox.combinatorial_search.RandomDimensionalSearch import RandomDimensionalSearch
from dsbox.combinatorial_search.BanditDimensionalSearch import BanditDimensionalSearch
from dsbox.combinatorial_search.MultiBanditSearch import MultiBanditSearch

from dsbox.combinatorial_search.search_utils import get_target_columns
from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.template.template import DSBoxTemplate
from common_primitives import utils as common_primitives_utils

import dsbox.JobManager.mplog as mplog

__all__ = ['Status', 'Controller']

import copy
import pprint
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# FIXME: we only need this for testing
import pandas as pd

FILE_FORMATTER = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
FILE_LOGGING_LEVEL = logging.INFO
LOG_FILENAME = 'dsbox.log'
CONSOLE_LOGGING_LEVEL = logging.INFO
# CONSOLE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_FORMATTER = "[%(levelname)s] - %(name)s - %(message)s"

pd.set_option("display.max_rows", 100)

class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148


class Controller:
    TIMEOUT = 59  # in minutes

    def __init__(self, development_mode: bool = False, run_single_template_name: str = "") -> None:
        self.development_mode: bool = development_mode

        self.run_single_template_name = run_single_template_name

        # Do not use, should use parsed results from key/value pairs of config
        # TA3 API may not provid the same information
        # self.config: typing.Dict = {}

        # Problem
        self.problem: typing.Dict = {}
        self.task_type: TaskType = None
        self.task_subtype: TaskSubtype = None
        self.problem_doc_metadata: Metadata = None
        self.problem_info = {}

        # Dataset
        self.dataset_schema_file: str = ""
        self.train_dataset1: Dataset = None
        self.train_dataset2: typing.List[Dataset] = None
        self.test_dataset1: Dataset = None
        self.test_dataset2: typing.List[Dataset] = None
        self.all_dataset: Dataset = None
        self.taskSourceType: typing.Set[str] = set()  # str from SEMANTIC_TYPES

        # Dataset limits
        self.threshold_column_length = 300
        self.threshold_index_length = 100000
        # hard coded unsplit dataset type
        # TODO: check whether "speech" type should be put into this list or not
        self.data_type_cannot_split = ["graph", "edgeList", "audio"]
        self.task_type_can_split = ["CLASSIFICATION", "REGRESSION", "TIME_SERIES_FORECASTING"]

        # Resource limits
        self.num_cpus: int = 0
        self.ram: int = 0  # concurrently ignored
        self.timeout: int = 0  # in seconds

        # Templates
        if self.run_single_template_name:
            self.template_library = TemplateLibrary(run_single_template=run_single_template_name)
        else:
            self.template_library = TemplateLibrary()
        self.template: typing.List[DSBoxTemplate] = []
        self.max_split_times = 1

        # Primitives
        self.primitive: typing.Dict = d3m.index.search()

        # set random seed
        random.seed(4676)

        # Output directories
        self.output_directory: str = '/output/'
        self.output_pipelines_dir: str = ""
        self.output_executables_dir: str = ""
        self.output_supporting_files_dir: str = ""
        self.output_logs_dir: str = ""
        self._logger = None

        self.main_pid: int = os.getpid()

    '''
        **********************************************************************
        Private method
        1. _check_and_set_dataset_metadata
        2. _create_output_directory
        3. _load_schema
        4. _log_init
        5. _log_search_results
        6. _process_pipeline_submission
        7. _run_BanditDimSearch
        8. _run_ParallelBaseSearch
        9. _run_RandomDimSearch
        10.
        **********************************************************************
    '''

    def _check_and_set_dataset_metadata(self) -> None:
        # Need to make sure the Target and TrueTarget column semantic types are set
        if self.task_type == TaskType.CLASSIFICATION or self.task_type == TaskType.REGRESSION:

            # start from last column, since typically target is the last column
            for index in range(
                    self.all_dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length'] - 1,
                    -1, -1):
                column_semantic_types = self.all_dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if ('https://metadata.datadrivendiscovery.org/types/Target' in column_semantic_types
                        and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in
                        column_semantic_types):
                    return

            # If not set, use sugested target column
            for index in range(
                    self.all_dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length'] - 1,
                    -1, -1):
                column_semantic_types = self.all_dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in \
                        column_semantic_types:
                    column_semantic_types = list(column_semantic_types) + [
                        'https://metadata.datadrivendiscovery.org/types/Target',
                        'https://metadata.datadrivendiscovery.org/types/TrueTarget']
                    self.all_dataset.metadata = self.all_dataset.metadata.update(
                        ('0', ALL_ELEMENTS, index), {'semantic_types': column_semantic_types})
                    return

            raise InvalidArgumentValueError(
                'At least one column should have semantic type SuggestedTarget')

    def _create_output_directory(self, config):
        '''
        Create output sub-directories based on Summer 2018 evaluation layout.

        For the Summer 2018 evaluation the top-level output dir is '/output'
        '''
        #### Official config entry for Evaluation
        if 'pipeline_logs_root' in config:
            self.output_pipelines_dir = os.path.abspath(config['pipeline_logs_root'])
        if 'executables_root' in config:
            self.output_executables_dir = os.path.abspath(config['executables_root'])
        if 'temp_storage_root' in config:
            self.output_supporting_files_dir = os.path.abspath(config['temp_storage_root'])
        #### End: Official config entry for Evaluation

        if 'D3MOUTPUTDIR' in os.environ:
            self.output_directory = os.path.abspath(os.environ['D3MOUTPUTDIR'])
        else:
            self.output_directory = os.path.split(self.output_executables_dir)[0]

        if 'logs_root' in config:
            self.output_logs_dir = os.path.abspath(config['logs_root'])
        else:
            self.output_logs_dir = os.path.join(self.output_supporting_files_dir, 'logs')

        # Make directories if they do not exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        for path in [self.output_pipelines_dir, self.output_executables_dir,
                     self.output_supporting_files_dir, self.output_logs_dir]:
            if not os.path.exists(path) and path != '':
                os.makedirs(path)

        self._log_init()
        self._logger.info('Top level output directory: %s' % self.output_directory)
        considered_root = os.path.join(os.path.dirname(self.output_pipelines_dir),
                                       'pipelines_considered')
        self._logger.info('Considered output directory: %s' % considered_root)

    def _load_schema(self, config):
        # Do not use
        # self.config = config

        # Problem
        self.problem = parse_problem_description(config['problem_schema'])
        self.problem_doc_metadata = runtime.load_problem_doc(
            os.path.abspath(config['problem_schema']))
        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']

        # Dataset
        self.dataset_schema_file = config['dataset_schema']

        # find the data resources type
        self.taskSourceType = set()  # set the type to be set so that we can ignore the repeat
        # elements
        with open(self.dataset_schema_file, 'r') as dataset_description_file:
            dataset_description = json.load(dataset_description_file)
            for each_type in dataset_description["dataResources"]:
                self.taskSourceType.add(each_type["resType"])
        self.problem_info["data_type"] = self.taskSourceType

        # Resource limits
        self.num_cpus = int(config.get('cpus', 0))
        self.ram = config.get('ram', 0)
        self.timeout = (config.get('timeout', self.TIMEOUT)) * 60
        self.saved_pipeline_id = config.get('saved_pipeline_ID', "")

        # def _generate_problem_info(self,problem):
        for i in range(len(self.problem['inputs'])):
            if 'targets' in self.problem['inputs'][i]:
                break

        self.problem_info["task_type"] = self.problem['problem'][
            'task_type'].name  # 'classification' 'regression'
        self.problem_info["res_id"] = self.problem['inputs'][i]['targets'][0]['resource_id']
        self.problem_info["target_index"] = []
        for each in self.problem['inputs'][i]['targets']:
            self.problem_info["target_index"].append(each["column_index"])

    def _log_init(self) -> None:
        logging.basicConfig(
            level=FILE_LOGGING_LEVEL,
            format=FILE_FORMATTER,
            datefmt='%m-%d %H:%M:%S',
            filename=os.path.join(self.output_logs_dir, LOG_FILENAME),
            filemode='w'
        )

        self._logger = logging.getLogger(__name__)

        if self._logger.getEffectiveLevel() <= 10:
            os.makedirs(os.path.join(self.output_logs_dir, "dfs"), exist_ok=True)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(CONSOLE_FORMATTER))
        console.setLevel(CONSOLE_LOGGING_LEVEL)
        self._logger.addHandler(console)

    def _log_search_results(self, report: typing.Dict[str, typing.Any]):
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
                if m in report and report[m]:
                    self._logger.info('{} {} = {}'.format(
                        m, report[m][0]['metric'], report[m][0]['value']))

        # # writing to the disk
        # dataset_name = self.output_executables_dir.rsplit("/", 2)[1]
        # # save_location = os.path.join(self.output_logs_dir, dataset_name + ".txt")
        # save_location = self.output_directory + ".txt"
        #
        # self._logger.info("******************\n[INFO] Saving training results in %s", save_location)
        # metrics = self.problem['problem']['performance_metrics']
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
        output_dir = os.path.dirname(self.output_pipelines_dir)
        print("[PROSKA]:", output_dir)
        pipelines_root: str = os.path.join(output_dir, 'pipelines')
        executables_root: str = os.path.join(output_dir, 'executables')
        supporting_root: str = os.path.join(output_dir, 'supporting_files')
        # os.path.join(os.path.dirname(executables_root), 'pipelines')

        # Read all the json files in the pipelines
        piplines_name_list = os.listdir(pipelines_root)
        if len(piplines_name_list) < 20:
            return

        pipelines_df = pd.DataFrame(0.0, index=piplines_name_list, columns=["rank"])
        for name in piplines_name_list:
            with open(os.path.join(pipelines_root, name)) as f:
                try:
                    rank = json.load(f)['pipeline_rank']
                except (json.decoder.JSONDecodeError, KeyError) as e:
                    rank = 0
            pipelines_df.at[name, 'rank'] = rank

        # sort them based on their rank field
        pipelines_df.sort_values(by='rank', ascending=True, inplace=True)

        # make sure that "pipeline_considered" directory exists
        considered_root = os.path.join(os.path.dirname(pipelines_root), 'pipelines_considered')
        try:
            os.mkdir(considered_root)
        except FileExistsError:
            pass

        # pick the top 20 and move the rest to "pipeline_considered" directory
        for name in pipelines_df.index[20:]:
            os.rename(src=os.path.join(pipelines_root, name),
                      dst=os.path.join(considered_root, name))

        # delete the exec and supporting files related the moved pipelines
        for name in pipelines_df.index[20:]:
            pipeName = name.split('.')[0]
            try:
                os.remove(os.path.join(executables_root, pipeName + '.json'))
            except FileNotFoundError:
                traceback.print_exc()
                pass

            try:
                shutil.rmtree(os.path.join(supporting_root, pipeName))
            except FileNotFoundError:
                traceback.print_exc()
                pass

    def _run_SerialBaseSearch(self):
        searchMethod = TemplateSpaceBaseSearch(
            template_list=self.template,
            performance_metrics=self.problem['problem']['performance_metrics'],
            problem=self.problem_doc_metadata,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            output_directory=self.output_directory,
            log_dir=self.output_logs_dir,
        )
        report = searchMethod.search(num_iter=10)

        self._log_search_results(report=report)

    def _run_ParallelBaseSearch(self):
        searchMethod = TemplateSpaceParallelBaseSearch(
            template_list=self.template,
            performance_metrics=self.problem['problem']['performance_metrics'],
            problem=self.problem_doc_metadata,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            output_directory=self.output_directory,
            log_dir=self.output_logs_dir,
            num_proc=self.num_cpus,
            timeout=self.TIMEOUT,
        )
        report = searchMethod.search(num_iter=15)

        self._log_search_results(report=report)

        searchMethod.job_manager.kill_job_mananger()

    def _run_RandomDimSearch(self):
        searchMethod = RandomDimensionalSearch(
            template_list=self.template,
            performance_metrics=self.problem['problem']['performance_metrics'],
            problem=self.problem_doc_metadata,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            output_directory=self.output_directory,
            log_dir=self.output_logs_dir,
            num_proc=self.num_cpus,
            timeout=self.TIMEOUT,
        )
        report = searchMethod.search(num_iter=10)

        self._log_search_results(report=report)

        searchMethod.job_manager.kill_job_mananger()

    def _run_BanditDimSearch(self):
        searchMethod = BanditDimensionalSearch(
            template_list=self.template,
            performance_metrics=self.problem['problem']['performance_metrics'],
            problem=self.problem_doc_metadata,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            output_directory=self.output_directory,
            log_dir=self.output_logs_dir,
            num_proc=self.num_cpus,
            timeout=self.TIMEOUT,
        )
        report = searchMethod.search(num_iter=5)

        self._log_search_results(report=report)

        searchMethod.job_manager.kill_job_mananger()

    def _run_MultiBanditSearch(self):
        searchMethod = MultiBanditSearch(
            template_list=self.template,
            performance_metrics=self.problem['problem']['performance_metrics'],
            problem=self.problem_doc_metadata,
            test_dataset1=self.test_dataset1,
            train_dataset1=self.train_dataset1,
            test_dataset2=self.test_dataset2,
            train_dataset2=self.train_dataset2,
            all_dataset=self.all_dataset,
            output_directory=self.output_directory,
            log_dir=self.output_logs_dir,
            num_proc=self.num_cpus,
            timeout=self.TIMEOUT,
        )
        report = searchMethod.search(num_iter=30)

        self._log_search_results(report=report)

        searchMethod.job_manager.kill_job_mananger()
    '''
        **********************************************************************
        Public method (in alphabet)
        1 . auto_regress_convert
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
        **********************************************************************
    '''

    def auto_regress_convert(self, dataset: "Dataset"):
        '''
        do auto convert for timeseriesforecasting prob
        '''
        problem = self.problem_doc_metadata.query(())
        targets = problem["inputs"]["data"][0]["targets"]
        for each_target in range(len(targets)):
            resID = targets[each_target]["resID"]
            colIndex = targets[each_target]["colIndex"]
            if problem["about"]["taskType"] == "timeSeriesForecasting" or problem["about"][
                "taskType"] == "regression":
                dataset[resID].iloc[:, colIndex] = pd.to_numeric(dataset[resID].iloc[:, colIndex],
                                                                 downcast="float", errors="coerce")
                meta = dict(dataset.metadata.query((resID, ALL_ELEMENTS, colIndex)))
                meta["structural_type"] = float
                dataset.metadata = dataset.metadata.update((resID, ALL_ELEMENTS, colIndex), meta)
        return dataset

    @staticmethod
    def generate_configuration_space(template_desc: TemplateDescription, problem: typing.Dict,
                                     dataset: typing.Optional[Dataset]) -> ConfigurationSpace:
        """
        Generate search space
        """

        # TODO: Need to update dsbox.planner.common.ontology.D3MOntology and
        # dsbox.planner.common.ontology.D3MPrimitiveLibrary, and integrate with them
        libdir = os.path.join(os.getcwd(), "library")
        # print(libdir)
        mapper_to_primitives = SemanticTypeDict(libdir)
        mapper_to_primitives.read_primitives()
        # print(mapper_to_primitives.mapper)
        # print(mapper_to_primitives.mapper)
        values = mapper_to_primitives.create_configuration_space(template_desc.template)
        # print(template_desc.template.template_nodes.items())
        print("[INFO] Values: {}".format(values))
        # values: typing.Dict[DimensionName, typing.List] = {}
        return SimpleConfigurationSpace(values)

    def initialize_from_config_for_evaluation(self, config: typing.Dict) -> None:
        '''
            This function for running ta2_evaluation
        '''
        self._load_schema(config)
        self._create_output_directory(config)

        # Dataset
        loader = D3MDatasetLoader()
        json_file = os.path.abspath(self.dataset_schema_file)
        all_dataset_uri = 'file://{}'.format(json_file)
        self.all_dataset = loader.load(dataset_uri=all_dataset_uri)

        # load templates
        self.load_templates()

    def initialize_from_config_train_test(self, config: typing.Dict) -> None:
        '''
            This function for running for ta2-search
        '''
        self._load_schema(config)
        self._create_output_directory(config)

        # Dataset
        loader = D3MDatasetLoader()

        json_file = os.path.abspath(self.dataset_schema_file)
        all_dataset_uri = 'file://{}'.format(json_file)
        self.all_dataset = loader.load(dataset_uri=all_dataset_uri)

        # Templates
        self.load_templates()

    def load_pipe_runtime(self):
        d = os.path.expanduser(self.output_directory + '/pipelines')
        read_pipeline_id = self.saved_pipeline_id
        if read_pipeline_id == "":
            self._logger.info(
                "[INFO] No specified pipeline ID found, will load the latest "
                "crated pipeline.")
            # if no pipeline ID given, load the newest created file in the
            # folder
            files = [os.path.join(d, f) for f in os.listdir(d)]
            files.sort(key=lambda f: os.stat(f).st_mtime)
            lastmodified = files[-1]
            read_pipeline_id = lastmodified.split('/')[-1].split('.')[0]

        pipeline_load, run = FittedPipeline.load(folder_loc=self.output_directory,
                                                 pipeline_id=read_pipeline_id,
                                                 log_dir=self.output_logs_dir)
        return self.output_directory, pipeline_load, read_pipeline_id, run

    def load_templates(self) -> None:
        self.template = self.template_library.get_templates(self.task_type, self.task_subtype,
                                                            self.taskSourceType)
        # find the maximum dataset split requirements
        for each_template in self.template:
            for each_step in each_template.template['steps']:
                if "runtime" in each_step and "test_validation" in each_step["runtime"]:
                    split_times = int(each_step["runtime"]["test_validation"])
                    if split_times > self.max_split_times:
                        self.max_split_times = split_times

    def remove_empty_targets(self, dataset: "Dataset"):
        '''
        will automatically remove empty targets in training
        '''
        problem = self.problem_doc_metadata.query(())
        targets = problem["inputs"]["data"][0]["targets"]
        resID = targets[0]["resID"]
        colIndex = targets[0]["colIndex"]
        # dataset_actual = dataset[resID]

        droplist = []
        for i, v in dataset[resID].iterrows():
            if v[colIndex] == "":
                droplist.append(i)
        if droplist != []:
            dataset[resID] = dataset[resID].drop(dataset[resID].index[droplist])
            meta = dict(dataset.metadata.query((resID,)))
            dimension = dict(meta['dimension'])
            meta['dimension'] = dimension
            dimension['length'] = dataset[resID].shape[0]
            dataset.metadata = dataset.metadata.update((resID,), meta)

        return dataset

    def split_dataset(self, dataset, random_state=42, test_size=0.2, n_splits=1,
                      need_test_dataset=True):
        '''
            Split dataset into 2 parts for training and test
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

        task_type = self.problem_info[
            "task_type"]  # ['problem']['task_type'].name  # 'classification' 'regression'
        res_id = self.problem_info["res_id"]
        target_index = self.problem_info["target_index"]
        data_type = self.problem_info["data_type"]

        train_return = []
        test_return = []

        cannot_split = False

        for each in data_type:
            if each in self.data_type_cannot_split:
                cannot_split = True
                break

        # check second time if the program think we still can split
        if not cannot_split:
            if task_type is not list:
                task_type_check = [task_type]

            for each in task_type_check:
                if each not in self.task_type_can_split:
                    cannot_split = True
                    break

        # if the dataset type in the list that we should not split
        if cannot_split:
            for i in range(n_splits):
                # just return all dataset to train part
                train_return.append(dataset)
                test_return.append(None)

        # if the dataset type can be split
        else:
            if task_type == 'CLASSIFICATION':
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

                        outdf_train = d3m_DataFrame(outdf_train, generate_metadata=False)
                        outdf_train = outdf_train.set_index("d3mIndex", drop=False)
                        train = _add_meta_data(dataset=dataset, res_id=res_id,
                                               input_part=outdf_train)
                        # train = _add_meta_data(dataset = dataset, res_id = res_id, input_part =
                        #  dataset[res_id].iloc[train_index, :])
                        train_return.append(train)

                        # for special condition that only need get part of the dataset
                        if need_test_dataset:
                            outdf_test = pd.DataFrame(columns=dataset[res_id].columns)
                            for each_index in test_index:
                                outdf_test = outdf_test.append(indf.loc[each_index],
                                                               ignore_index=True)
                            outdf_test = d3m_DataFrame(outdf_test, generate_metadata=False)
                            outdf_test = outdf_test.set_index("d3mIndex", drop=False)
                            test = _add_meta_data(dataset=dataset, res_id=res_id,
                                                  input_part=outdf_test)
                            # test = _add_meta_data(dataset = dataset, res_id = res_id,
                            # input_part = dataset[res_id].iloc[test_index, :])
                            test_return.append(test)
                        else:
                            test_return.append(None)
                except:
                    # Do not split stratified shuffle fails
                    print('[Info] Not splitting dataset. Stratified shuffle failed')
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
                                           input_part=dataset[res_id].iloc[train_index, :])
                    train_return.append(train)
                    # for special condition that only need get part of the dataset
                    if need_test_dataset:
                        test = _add_meta_data(dataset=dataset, res_id=res_id,
                                              input_part=dataset[res_id].iloc[test_index, :])
                        test_return.append(test)
                    else:
                        test_return.append(None)

        return (train_return, test_return)

    def test(self) -> Status:
        """
            First read the fitted pipeline and then run trained pipeline on test data.
        """
        self._logger.info("[INFO] Start test function")
        outputs_loc, pipeline_load, read_pipeline_id, run_test = self.load_pipe_runtime()

        self._logger.info("[INFO] Pipeline load finished")

        self._logger.info("[INFO] testing data")

        self.all_dataset = self.auto_regress_convert(self.all_dataset)
        runtime.add_target_columns_metadata(self.all_dataset, self.problem_doc_metadata)
        run_test.produce(inputs=[self.all_dataset])

        try:
            step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        except:
            self._logger.error("Warning: searching the output step number failed! "
                               "Will use the last step's output of the pipeline.")
            # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            step_number_output = len(run_test.produce_outputs) - 1

        # get the target column name
        prediction_class_name = []
        try:
            with open(self.dataset_schema_file, 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_resource in dataset_description["dataResources"]:
                    if "columns" in each_resource:
                        for each_column in each_resource["columns"]:
                            if "suggestedTarget" in each_column["role"] or "target" in each_column[
                                "role"]:
                                prediction_class_name.append(each_column["colName"])
        except:
            self._logger.error(
                "[Warning] Can't find the prediction class name, will use default name "
                "'prediction'.")
            prediction_class_name.append("prediction")

        prediction = run_test.produce_outputs[step_number_output]

        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            # prediction.drop("confidence", axis=1, inplace=True, errors = "ignore")#some
            # prediction has "confidence"
            prediction_col_name = ['d3mIndex']
            for each in prediction.columns:
                prediction_col_name.append(each)
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[prediction_col_name]
            prediction_col_name.remove('d3mIndex')
            for i in range(len(prediction_class_name)):
                prediction = prediction.rename(
                    columns={prediction_col_name[i]: prediction_class_name[i]})
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
        d = os.path.expanduser(self.output_directory + '/pipelines')
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

        pipeline_load, run_test = FittedPipeline.load(folder_loc=self.output_directory,
                                                      pipeline_id=fitted_pipeline_id,
                                                      log_dir=self.output_logs_dir)

        print("[INFO] Pipeline load finished")

        print("[INFO] testing data:")
        # pprint(self.test_dataset.head())

        # pipeline_load.runtime.produce(inputs=[self.test_dataset])
        self.all_dataset = self.auto_regress_convert(self.all_dataset)
        runtime.add_target_columns_metadata(self.all_dataset, self.problem_doc_metadata)
        run_test.produce(inputs=[self.all_dataset])

        try:
            step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        except:
            self._logger.error("Warning: searching the output step number failed! "
                               "Will use the last step's output of the pipeline.")
            # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            step_number_output = len(run_test.produce_outputs) - 1

        # get the target column name
        prediction_class_name = []
        try:
            with open(self.dataset_schema_file, 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_resource in dataset_description["dataResources"]:
                    if "columns" in each_resource:
                        for each_column in each_resource["columns"]:
                            if "suggestedTarget" in each_column["role"] or "target" in each_column[
                                "role"]:
                                prediction_class_name.append(each_column["colName"])
        except:
            self._logger.error(
                "[Warning] Can't find the prediction class name, will use default name "
                "'prediction'.")
            prediction_class_name.append("prediction")

        prediction = run_test.produce_outputs[step_number_output]
        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            # prediction.drop("confidence", axis=1, inplace=True, errors = "ignore")#some
            # prediction has "confidence"
            prediction_col_name = ['d3mIndex']
            for each in prediction.columns:
                prediction_col_name.append(each)
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[prediction_col_name]
            prediction_col_name.remove('d3mIndex')
            for i in range(len(prediction_class_name)):
                prediction = prediction.rename(
                    columns={prediction_col_name[i]: prediction_class_name[i]})
        prediction_folder_loc = self.output_directory + "/predictions/" + fitted_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
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

    def train(self) -> Status:
        """
        Generate and train pipelines.
        """
        if not self.template:
            return Status.PROBLEM_NOT_IMPLEMENT

        # self._check_and_set_dataset_metadata()

        self.generate_dataset_splits()

        # FIXME) come up with a better way to implement this part. The fork does not provide a way
        # FIXME) to catch the errors of the child process
        with mplog.open_queue() as log_queue:
            self._logger.info('Starting Search process')

            proc = Process(target=mplog.logged_call,
                           args=(log_queue, self._run_MultiBanditSearch,))
            # proc = Process(target=mplog.logged_call,
            #                args=(log_queue, self._run_RandomDimSearch,))
            # proc = Process(target=mplog.logged_call,
            #                args=(log_queue, self._run_ParallelBaseSearch,))
            # proc = Process(target=mplog.logged_call,
            #                args=(log_queue, self._run_SerialBaseSearch,))
            # _run_RandomDimSearch
            proc.start()

            self._logger.info('Searching is finished')
            # wait until process is done
            proc.join()

            status = proc.exitcode
            print("[INFO] Search Status:")
            pprint.pprint(status)
        
        print("END OF FORK")


    def generate_dataset_splits(self):
        self.all_dataset = self.remove_empty_targets(self.all_dataset)
        self.all_dataset = self.auto_regress_convert(self.all_dataset)
        runtime.add_target_columns_metadata(self.all_dataset, self.problem_doc_metadata)
        res_id = self.problem_info['res_id']
        # check the shape of the dataset
        main_res_shape = self.all_dataset[res_id].shape
        # if the column length is larger than the threshold, it may failed in the given time, so we need to sample part of the dataset

        if main_res_shape[1] > self.threshold_column_length:
            self._logger.info(
                "The columns number of the input dataset is very large, now sampling part of them.")

            # first check the target column amount
            target_column_list = []
            all_column_length = \
            self.all_dataset.metadata.query((res_id, ALL_ELEMENTS))['dimension']['length']
            # for each_column in range(all_column_length - 1, 0, -1):
            #     each_column_meta = self.all_dataset.metadata.query((res_id,ALL_ELEMENTS,each_column))
            #     if ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget' or  'https://metadata.datadrivendiscovery.org/types/Target' or  'https://metadata.datadrivendiscovery.org/types/TrueTarget') in each_column_meta['semantic_types']:
            #         target_column_list.append(each_column)
            #     # to accelerate the program running, now we assume the target columns are always at the end of the columns
            # self._logger.info("Totally {} taget found.".format(len(target_column_list)))
            # target_column_length = len(target_column_list)

            # changed to use problem metadata to query targets
            targets_from_problem = self.problem_doc_metadata.query(())["inputs"]["data"][0][
                "targets"]
            for t in targets_from_problem:
                target_column_list.append(t["colIndex"])
            self._logger.info("Totally {} taget found.".format(len(target_column_list)))
            target_column_length = len(target_column_list)
            # check again on the length of the column to ensure
            if (main_res_shape[1] - target_column_length - 1) <= self.threshold_column_length:
                pass
            else:
                attribute_column_length = all_column_length - target_column_length - 1
                # skip the column 0 which is d3mIndex]
                is_all_numerical = True
                # check whether all inputs are categorical or not
                # for each_column in range(1, attribute_column_length + 1):
                #     each_metadata = self.all_dataset.metadata.query((res_id,ALL_ELEMENTS,each_column))
                #     if 'http://schema.org/Float' not in each_metadata['semantic_types'] or 'http://schema.org/Integer' not in each_metadata['semantic_types']:
                #         is_all_numerical = False
                #         break
                # two ways to do sampling (random projection or random choice)
                if is_all_numerical:
                    # TODO:
                    # add special template that use random projection directly
                    # add one special source type for the template special process such kind of dataset
                    self._logger.info(
                        "Special type of dataset: large column number with all categorical columns.")
                    self._logger.info("Will reload the template with new task source type.")
                    self.taskSourceType.add("large_column_number")
                    # aadd new template specially for large column numbers at the first priority
                    new_template = self.template_library.get_templates(self.task_type,
                                                                       self.task_subtype,
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
                    problem = dict(self.problem_doc_metadata.query(()))
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
                    self.problem_doc_metadata = self.problem_doc_metadata.update((), problem)
                    # updating problem_doc_metadata finished

                    self._logger.info("Random sampling on columns Finished.")

        if main_res_shape[0] > self.threshold_index_length:
            self._logger.info(
                "The row number of the input dataset is very large, will send only part of them to search.")
            if main_res_shape[1] > 20:
                self.threshold_index_length = int(self.threshold_index_length * 0.3)
                self._logger.info(
                    "The column number is also very large, will reduce the sampling amount on row number.")
            # too many indexs, we can run another split dataset
            index_removed_percent = 1 - float(self.threshold_index_length) / float(
                main_res_shape[0])
            # ignore the test part
            self.all_dataset, _ = self.split_dataset(dataset=self.all_dataset,
                                                     test_size=index_removed_percent,
                                                     need_test_dataset=False)
            self.all_dataset = self.all_dataset[0]
            self._logger.info("Random sampling on rows Finished.")

        # split the dataset first time
        self.train_dataset1, self.test_dataset1 = self.split_dataset(dataset=self.all_dataset)

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
            self.train_dataset2, self.test_dataset2 = self.split_dataset(
                dataset=self.train_dataset1, test_size=0.1, n_splits=self.max_split_times)
            if len(self.train_dataset2) < 1:
                self._logger.error(
                    "Some error happend with train_dataset1 split: The length of splitted dataset is less than 1")
            if len(self.test_dataset2) < 1:
                self._logger.error("Split failed on train_dataset1.")
                self.test_dataset2 = None
        else:
            self.train_dataset2 = None
            self.test_dataset2 = None