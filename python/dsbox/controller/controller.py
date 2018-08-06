import enum
import json
import logging
import os
import random
import typing
import uuid

from multiprocessing import Manager
from math import sqrt, log
import traceback
import importlib
import frozendict
spam_spec = importlib.util.find_spec("colorama")
STYLE = ""
ERROR = ""
WARNING = ""
if spam_spec is not None:
    from colorama import Fore, Back, init
    # STYLE = Fore.BLUE + Back.GREEN
    STYLE = Fore.BLACK + Back.GREEN
    ERROR = Fore.WHITE + Back.RED
    WARNING = Fore.BLACK + Back.YELLOW
    init(autoreset=True)

import pandas as pd
import d3m
import dsbox.template.runtime as runtime

from d3m.metadata.problem import TaskType
from d3m.container.pandas import DataFrame as d3m_DataFrame
from d3m.container.dataset import Dataset
from d3m.container.dataset import D3MDatasetLoader
from d3m.exceptions import NotSupportedError
from d3m.exceptions import InvalidArgumentValueError
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.base import Metadata
from d3m.metadata.problem import TaskSubtype
from d3m.metadata.problem import parse_problem_description

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.template.library import TemplateDescription
from dsbox.template.library import TemplateLibrary
from dsbox.template.library import SemanticTypeDict
from dsbox.template.search import ConfigurationSpace
from dsbox.template.search import SimpleConfigurationSpace
from dsbox.template.search import TemplateDimensionalSearch
from dsbox.template.search import get_target_columns, random_choices_without_replacement
from dsbox.template.template import DSBoxTemplate
from common_primitives import utils as common_primitives_utils

__all__ = ['Status', 'Controller']

import copy
import pprint
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# FIXME: we only need this for testing
import pandas as pd

FILE_FORMATTER = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
FILE_LOGGING_LEVEL = logging.DEBUG
LOG_FILENAME = 'dsbox.log'
CONSOLE_LOGGING_LEVEL = logging.INFO
# CONSOLE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_FORMATTER = "[%(levelname)s] - %(name)s - %(message)s"

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
        self.data_type_cannot_split = ["graph","edgeList", "audio"]
        self.task_type_can_split = ["CLASSIFICATION","REGRESSION"]

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

    '''
        **********************************************************************
        Private method
        1. _check_and_set_dataset_metadata
        2. _create_output_directory
        3. _load_schema
        4. _log_init
        **********************************************************************
    '''
    def _check_and_set_dataset_metadata(self) -> None:
        # Need to make sure the Target and TrueTarget column semantic types are set
        if self.task_type == TaskType.CLASSIFICATION or self.task_type == TaskType.REGRESSION:

            # start from last column, since typically target is the last column
            for index in range(self.all_dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length']-1, -1, -1):
                column_semantic_types = self.all_dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if ('https://metadata.datadrivendiscovery.org/types/Target' in column_semantic_types
                        and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in column_semantic_types):
                    return

            # If not set, use sugested target column
            for index in range(self.all_dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length']-1, -1, -1):
                column_semantic_types = self.all_dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in column_semantic_types:
                    column_semantic_types = list(column_semantic_types) + ['https://metadata.datadrivendiscovery.org/types/Target',
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

    def _load_schema(self, config):
        # Do not use
        # self.config = config

        # Problem
        self.problem = parse_problem_description(config['problem_schema'])
        self.problem_doc_metadata = runtime.load_problem_doc(os.path.abspath(config['problem_schema']))
        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']

        # Dataset
        self.dataset_schema_file = config['dataset_schema']

        # find the data resources type
        self.taskSourceType = set()  # set the type to be set so that we can ignore the repeat elements
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

        self.problem_info["task_type"] = self.problem['problem']['task_type'].name  # 'classification' 'regression'
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

    '''
        **********************************************************************
        Public method (in alphabet)
        1 . auto_regress_convert
        2 . compute_UCT
        3 . generate_configuration_space
        4 . initialize_from_config_for_evaluation
        5 . initialize_from_config_train_test
        6 . initialize_uct
        7 . load_pipe_runtime
        8 . load_templates
        9 . remove_empty_targets
        10. search_template
        11. select_next_template
        12. split_dataset
        13. test
        14. test_fitted_pipeline
        15. train
        16. update_history
        17. update_UCT_score
        18. write_training_results
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
            if problem["about"]["taskType"] == "timeSeriesForecasting" or problem["about"]["taskType"] == "regression":
                dataset[resID].iloc[:, colIndex] = pd.to_numeric(dataset[resID].iloc[:, colIndex], downcast = "float", errors = "coerce")
                meta = dict(dataset.metadata.query((resID, ALL_ELEMENTS, colIndex)))
                meta["structural_type"] = float
                dataset.metadata = dataset.metadata.update((resID, ALL_ELEMENTS, colIndex), meta)
        return dataset

    def compute_UCT(self, index=0):
        beta = 10
        gamma = 1
        delta = 4
        history = self.normalize.iloc[index]
        try:

            reward = history['reward']
            # / history['trial']

            return (beta * (reward) * max(log(10*history['trial']), 1) +
                gamma * sqrt(2 * log(self.total_run) / history['trial']) +
                delta * sqrt(2 * log(self.total_time) / history['exe_time']))
        except:
            self._logger.error('Failed to compute UCT. Defaulting to 100.0')
            # print(STYLE+"[WARN] compute UCT failed:", history.tolist())
            return None


    @staticmethod
    def generate_configuration_space(template_desc: TemplateDescription, problem: typing.Dict,
                                     dataset: typing.Optional[Dataset]) -> ConfigurationSpace:
        """
        Generate search space
        """

        # TODO: Need to update dsbox.planner.common.ontology.D3MOntology and dsbox.planner.common.ontology.D3MPrimitiveLibrary, and integrate with them
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


    def initialize_uct(self):
        self.total_run = 0
        self.total_time = 0
        # self.exec_history = \
        #     [{"exe_time": 1, "reward": 1, "trial": 1, "candidate": None, "best_value": 0}] * \
        #     len(self.template)

        self.exec_history = pd.DataFrame(None,
                                         index=map(lambda s: s.template["name"], self.template),
                                         columns=['reward', 'exe_time', 'trial', 'candidate', 'best_value'])
        self.exec_history[['reward', 'exe_time', 'trial']] = 0
        self.exec_history[['best_value']] = float('-inf')

        self.exec_history['candidate'] = self.exec_history['candidate'].astype(object)
        self.exec_history['candidate'] = None

        # print(self.exec_history.to_string())
        self.uct_score = [None] * len(self.template)


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
        self.template = self.template_library.get_templates(self.task_type, self.task_subtype, self.taskSourceType)
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


    def search_template(self, template: DSBoxTemplate, candidate: typing.Dict=None,
                        cache_bundle: typing.Tuple[typing.Dict, typing.Dict]=(None, None)) \
            -> typing.Dict:

        self._logger.info('Searching template %s', template.template['name'])

        space = template.generate_configuration_space()

        metrics = self.problem['problem']['performance_metrics']

        # setup the dimensional search configs
        search = TemplateDimensionalSearch(
            template = template, configuration_space = space, problem = self.problem_doc_metadata,
            test_dataset1 = self.test_dataset1, train_dataset1 = self.train_dataset1,
            test_dataset2 = self.test_dataset2, train_dataset2 = self.train_dataset2,
            all_dataset = self.all_dataset, performance_metrics = metrics,
            output_directory=self.output_directory, log_dir=self.output_logs_dir,
            num_workers=self.num_cpus
            )

        self.minimize = search.minimize
        # candidate, value = search.search_one_iter()
        self._logger.info('cache size = {}'.format(len(cache_bundle[0])))
        report = search.search_one_iter(candidate_in=candidate, cache_bundle=cache_bundle)
        candidate = report['candidate']
        value = report['best_val']
        # assert "fitted_pipe" in candidate, "argument error!"
        if candidate is None:
            self._logger.error("[ERROR] not candidate!")
            return report  # return Status.PROBLEM_NOT_IMPLEMENT
        else:
            self._logger.info("******************\n[INFO] Writing results")
            pprint.pprint(candidate.data)
            self._logger.info(str(candidate.data)+ " "+ str(value))

            if candidate.data['training_metrics']:
                if type(candidate.data['training_metrics'][0]) is dict:
                    self._logger.info('Training {} = {}'.format(
                        candidate.data['training_metrics'][0]['metric'],
                        candidate.data['training_metrics'][0]['value']))
                else:
                    for each in candidate.data['training_metrics'][0]:
                        self._logger.info('Training  {} in {} = {}'.format(
                            each['column_name'],
                            each['metric'],
                            each['value']))
            if candidate.data['cross_validation_metrics']:
                if type(candidate.data['cross_validation_metrics'][0]) is dict:
                    self._logger.info('CV {} = {}'.format(
                        candidate.data['cross_validation_metrics'][0]['metric'],
                        candidate.data['cross_validation_metrics'][0]['value']))
                else:
                    for each in candidate.data['cross_validation_metrics'][0]:
                        self._logger.info('CV  {} in {} = {}'.format(
                            each['column_name'],
                            each['metric'],
                            each['value']))
            if candidate.data['test_metrics']:
                if type(candidate.data['test_metrics'][0]) is dict:
                    self._logger.info('Validation {} = {}'.format(
                        candidate.data['test_metrics'][0]['metric'],
                        candidate.data['test_metrics'][0]['value']))
                else:
                    for each in candidate.data['test_metrics'][0]:
                        self._logger.info('Validation of {} in {} = {}'.format(
                            each['column_name'],
                            each['metric'],
                            each['value']))

            return report

    def select_next_template(self, max_iter: int = 2):
        # while True:
        choices = list(range(len(self.template)))

        # initial evaluation
        for i in choices:
            yield i

        # print("[INFO] Choices:", choices)
        # UCT based evaluation
        # for i in range(max_iter):
        while True:
            valids = list(filter(lambda t: t[1] is not None,
                                 zip(choices, self.uct_score)))
            _choices = list(map(lambda t: t[0], valids))
            _weights = list(map(lambda t: t[1], valids))
            selected = random_choices_without_replacement(_choices, _weights, 1)
            yield selected[0]

    def split_dataset(self, dataset, random_state=42, test_size=0.2, n_splits=1, need_test_dataset = True):
        '''
            Split dataset into 2 parts for training and test
        '''
        def _add_meta_data(dataset, res_id, input_part):
            dataset_with_new_meta = copy.copy(dataset)
            dataset_metadata = dict(dataset_with_new_meta.metadata.query(()))
            dataset_metadata['id'] = dataset_metadata['id'] + '_' + str(uuid.uuid4())
            dataset_with_new_meta.metadata = dataset_with_new_meta.metadata.update((), dataset_metadata)

            dataset_with_new_meta[res_id] = input_part
            meta = dict(dataset_with_new_meta.metadata.query((res_id,)))
            dimension = dict(meta['dimension'])
            meta['dimension'] = dimension
            dimension['length'] = input_part.shape[0]
            # print(meta)
            dataset_with_new_meta.metadata = dataset_with_new_meta.metadata.update((res_id,), meta)
            # pprint(dict(dataset_with_new_meta.metadata.query((res_id,))))
            return dataset_with_new_meta

        task_type = self.problem_info["task_type"]  #['problem']['task_type'].name  # 'classification' 'regression'
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


        else:
            if task_type == 'CLASSIFICATION':
                try:
                    # Use stratified sample to split the dataset
                    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                    sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])
                    for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):
                        train = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[train_index, :])
                        train_return.append(train)
                        # for special condition that only need get part of the dataset
                        if need_test_dataset:
                            test = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[test_index, :])
                            test_return.append(test)
                        else:
                            test_return.append(None)
                except:
                    # Do not split stratified shuffle fails
                    print('[Info] Not splitting dataset. Stratified shuffle failed')
                    for i in range(n_splits):
                        train_return.append(dataset)
                        test_return.append(None)


                        
        # if the dataset type can be split
        # else:
        #     if task_type == 'CLASSIFICATION':
        #         try:
        #             # Use stratified sample to split the dataset
        #             sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        #             sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])
                    

        #             for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):

        #                 indf = dataset[res_id]
        #                 outdf_train = pd.DataFrame(columns = dataset[res_id].columns)
        #                 outdf_test = pd.DataFrame(columns = dataset[res_id].columns)
        #                 for each_index in train_index:
        #                     outdf_train = outdf_train.append(indf.loc[each_index],ignore_index = True)
        #                 for each_index in test_index:
        #                     outdf_test = outdf_test.append(indf.loc[each_index],ignore_index = True)
        #                 import pdb
        #                 pdb.set_trace()
        #                 outdf_train = d3m_DataFrame(outdf_train, generate_metadata = False)
        #                 outdf_test = d3m_DataFrame(outdf_test, generate_metadata = False)
        #                 outdf_train = outdf_train.set_index("d3mIndex", drop = False)
        #                 outdf_test = outdf_test.set_index("d3mIndex", drop = False)
        #                 train = _add_meta_data(dataset = dataset, res_id = res_id, input_part = outdf_train)
        #                 #train = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[train_index, :])
        #                 train_return.append(train)
        #                 # for special condition that only need get part of the dataset
        #                 if need_test_dataset:
        #                     test = _add_meta_data(dataset = dataset, res_id = res_id, input_part = outdf_test)
        #                     #test = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[test_index, :])
        #                     test_return.append(test)
        #                 else:
        #                     test_return.append(None)
        #         except:
        #             # Do not split stratified shuffle fails
        #             print('[Info] Not splitting dataset. Stratified shuffle failed')
        #             for i in range(n_splits):
        #                 train_return.append(dataset)
        #                 test_return.append(None)

            else:
                # Use random split
                if not task_type == "REGRESSION":

                    print('USING Random Split to split task type: {}'.format(task_type))
                ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                ss.get_n_splits(dataset[res_id])
                for train_index, test_index in ss.split(dataset[res_id]):
                    train = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[train_index, :])
                    train_return.append(train)
                    # for special condition that only need get part of the dataset
                    if need_test_dataset:
                        test = _add_meta_data(dataset = dataset, res_id = res_id, input_part = dataset[res_id].iloc[test_index, :])
                        test_return.append(test)
                    else:
                        test_return.append(None)

        return (train_return, test_return)


    def test(self) -> Status:
        """
            First read the fitted pipeline and then run trained pipeline on test data.
        """
        self._logger.info("[INFO] Start test function")
        outputs_loc, pipeline_load, read_pipeline_id, run_test = \
            self.load_pipe_runtime()

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
                            if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
                                prediction_class_name.append(each_column["colName"])
        except:
            self._logger.error("[Warning] Can't find the prediction class name, will use default name 'prediction'.")
            prediction_class_name.append("prediction")

        prediction = run_test.produce_outputs[step_number_output]
        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            prediction_col_name = ['d3mIndex']
            for each in prediction.columns:
                prediction_col_name.append(each)
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[prediction_col_name]
            prediction_col_name.remove('d3mIndex')
            for i in range(len(prediction_class_name)):
                prediction = prediction.rename(columns={prediction_col_name[i]: prediction_class_name[i]})
        prediction_folder_loc = outputs_loc + "/predictions/" + read_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)
        self._logger.info("[INFO] Finished: prediction results saving finished")
        self._logger.info("[INFO] The prediction results is stored at: {}".format(prediction_folder_loc))
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
                            if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
                                prediction_class_name.append(each_column["colName"])
        except:
            self._logger.error("[Warning] Can't find the prediction class name, will use default name 'prediction'.")
            prediction_class_name.append("prediction")

        prediction = run_test.produce_outputs[step_number_output]
        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            prediction_col_name = ['d3mIndex']
            for each in prediction.columns:
                prediction_col_name.append(each)
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[prediction_col_name]
            prediction_col_name.remove('d3mIndex')
            for i in range(len(prediction_class_name)):
                prediction = prediction.rename(columns={prediction_col_name[i]: prediction_class_name[i]})
        prediction_folder_loc = self.output_directory + "/predictions/" + fitted_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)
        self._logger.info("[INFO] Finished: prediction results saving finished")
        self._logger.info("[INFO] The prediction results is stored at: {}".format(prediction_folder_loc))
        return Status.OK


    def train(self) -> Status:
        """
            Generate and train pipelines.
        """
        if not self.template:
            return Status.PROBLEM_NOT_IMPLEMENT

        # self._check_and_set_dataset_metadata()

        self.initialize_uct()

        # setup the output cache
        manager = Manager()
        cache = manager.dict()
        candidate_cache = manager.dict()

        self.all_dataset = self.remove_empty_targets(self.all_dataset)
        self.all_dataset = self.auto_regress_convert(self.all_dataset)
        runtime.add_target_columns_metadata(self.all_dataset, self.problem_doc_metadata)
        res_id = self.problem_info['res_id']
        # check the shape of the dataset
        main_res_shape = self.all_dataset[res_id].shape
        # if the column length is larger than the threshold, it may failed in the given time, so we need to sample part of the dataset

        if main_res_shape[1] > self.threshold_column_length:
            self._logger.info("The columns number of the input dataset is very large, now sampling part of them.")
            
            # first check the target column amount
            target_column_list = []
            all_column_length = self.all_dataset.metadata.query((res_id,ALL_ELEMENTS))['dimension']['length']
            for each_column in range(all_column_length - 1, 0, -1):
                each_column_meta = self.all_dataset.metadata.query((res_id,ALL_ELEMENTS,each_column))
                if ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget' or  'https://metadata.datadrivendiscovery.org/types/Target' or  'https://metadata.datadrivendiscovery.org/types/TrueTarget') in each_column_meta['semantic_types']:
                    target_column_list.append(each_column)
                # to accelerate the program running, now we assume the target columns are always at the end of the columns
                else:
                    break
            self._logger.info("Totally {} taget found.".format(len(target_column_list)))
            target_column_length = len(target_column_list)
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
                self._logger.info("Special type of dataset: large column number with all categorical columns.")
                self._logger.info("Will reload the template with new task source type.")
                self.taskSourceType.add("large_column_number")
                # import pdb
                # pdb.set_trace()
                # # reload the template
                # new_template = self.template_library.get_templates(self.task_type, self.task_subtype, self.taskSourceType)
                # # find the maximum dataset split requirements
                # for each_template in new_template:
                #     for each_step in each_template.template['steps']:
                #         if "runtime" in each_step and "test_validation" in each_step["runtime"]:
                #             split_times = int(each_step["runtime"]["test_validation"])
                #             if split_times > self.max_split_times:
                #                 self.max_split_times = split_times
                # new_template = new_template.extend(self.template)
                self.load_templates()

            # else:
                # run sampling method to randomly throw some columns
                # update problem metadata
                problem = dict(self.problem_doc_metadata.query(()))
                #data_meta = dict(problem["inputs"]["data"][0])
                data_meta = []
                for each_data in problem["inputs"]["data"]:
                    # update targets metadata for each target columns
                    target_meta = []
                    each_data = dict(each_data)
                    for each_target in each_data["targets"]:
                        target_meta_each = dict(each_target)
                        target_meta_each['colIndex'] = self.threshold_column_length + (all_column_length - target_meta_each['colIndex'])
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
                self.problem_doc_metadata = self.problem_doc_metadata.update((),problem)
                # updating problem_doc_metadata finished

                all_attribute_columns = range(1, attribute_column_length + 1)
                # remove the old metadata which should not exist now
                # it should be done first, otherwise the remove operation will failed
                metadata_old = copy.copy(self.all_dataset.metadata)
                for each_removed_column in range(self.threshold_column_length + 1 + target_column_length, attribute_column_length + 1 + target_column_length):
                    self.all_dataset.metadata = self.all_dataset.metadata.remove((res_id,ALL_ELEMENTS, each_removed_column))
                # remove columns
                throw_columns = random.sample(all_attribute_columns, attribute_column_length - self.threshold_column_length)
                self.all_dataset[res_id] = common_primitives_utils.remove_columns(self.all_dataset[res_id], throw_columns, source='ISI DSBox')

                # update metadata on column information
                new_column_meta = dict(self.all_dataset.metadata.query((res_id,ALL_ELEMENTS)))
                new_column_meta['dimension'] = dict(new_column_meta['dimension'])
                new_column_meta['dimension']['length'] = self.threshold_column_length + 1 + target_column_length
                self.all_dataset.metadata = self.all_dataset.metadata.update((res_id,ALL_ELEMENTS),new_column_meta)

                # update the metadata on attribute column
                remained_columns = set(all_attribute_columns) - set(throw_columns)
                for new_column_count, each_remained_column in enumerate(remained_columns):
                    metadata_old_each = metadata_old.query((res_id,ALL_ELEMENTS, each_remained_column))
                    self.all_dataset.metadata = self.all_dataset.metadata.update((res_id,ALL_ELEMENTS, new_column_count + 1), metadata_old_each)

                # update class column
                for new_column_count, each_target_column in enumerate(target_column_list):
                    metadata_class = metadata_old.query((res_id,ALL_ELEMENTS, each_target_column))
                    self.all_dataset.metadata = self.all_dataset.metadata.update((res_id,ALL_ELEMENTS, self.threshold_column_length + new_column_count + 1), metadata_class)
                # update traget_index for spliting into train and test dataset
               
                if type(self.problem_info["target_index"]) is list:
                    for i in range(len(self.problem_info["target_index"])): 
                        self.problem_info["target_index"][i] = self.threshold_column_length + i + 1
                else:
                    self.problem_info["target_index"] = self.threshold_column_length + target_column_length

                self._logger.info("Random sampling on columns Finished.")

        if main_res_shape[0] > self.threshold_index_length:
            # too many indexs, we can run another split dataset
            self._logger.info("The indexs number of the input dataset is very large, will send only part of them to search.")
            index_removed_percent = 1 - float(self.threshold_index_length) / float(main_res_shape[0])
            # ignore the test part
            self.all_dataset, _ = self.split_dataset(dataset = self.all_dataset, test_size = index_removed_percent, need_test_dataset = False)
            self.all_dataset = self.all_dataset[0]

        # split the dataset first time
        self.train_dataset1, self.test_dataset1 = self.split_dataset(dataset = self.all_dataset)

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
            self.train_dataset2, self.test_dataset2 = self.split_dataset(dataset = self.train_dataset1, test_size = 0.1, n_splits = self.max_split_times)
            if len(self.train_dataset2) < 1:
                self._logger.error("Some error happend with train_dataset1 split: The length of splitted dataset is less than 1")
            if len(self.test_dataset2) < 1:
                self._logger.error("Split failed on train_dataset1.")
                self.test_dataset2 = None
        else:
            self.train_dataset2 = None
            self.test_dataset2 = None

        best_metric_value = None
        best_report = None
        self._logger.info("Preparing finished, now go to template search process.")

        for idx in self.select_next_template(max_iter=5):
            template = self.template[idx]
            self._logger.info(STYLE+"[INFO] Template {}:{} Selected. UCT:{}".format(idx, template.template['name'], self.uct_score))

            try:
                report = self.search_template(
                    template, candidate=self.exec_history.iloc[idx]['candidate'],
                    cache_bundle=(cache, candidate_cache),
                    )

            except:
                self._logger.exception("search_template failed on {} with UCT: {}".format(
                    template.template['name'], self.uct_score))
                traceback.print_exc()
                # report = {
                #
                # }
                continue
            self._logger.info(STYLE + "[INFO] report: " + str(report['best_val']))
            self.update_UCT_score(index=idx, report=report)
            self._logger.info(STYLE+"[INFO] cache size: " + str(len(cache))+
                  ", candidates: " + str(len(candidate_cache)))

            new_best = False
            if best_report is None:
                best_report = report
                best_metric_value = best_report['best_val']
                new_best = True
            else:
                if self.minimize and report['best_val'] < best_metric_value:
                    best_report = report
                    best_metric_value = report['best_val']
                    new_best = True
                if not self.minimize and report['best_val'] > best_metric_value:
                    best_report = report
                    best_metric_value = report['best_val']
                    new_best = True

            if new_best and best_report['candidate'] is not None:
                self._logger.info('[INFO] New Best Value: ' + str(report['best_val']))

                dataset_name = self.output_executables_dir.rsplit("/", 2)[1]
                # save_location = os.path.join(self.output_logs_dir, dataset_name + ".txt")
                save_location = self.output_directory + ".txt"

                self._logger.info("******************\n[INFO] Saving training results in %s", save_location)
                metrics = self.problem['problem']['performance_metrics']
                candidate = best_report['candidate']
                try:
                    f = open(save_location, "w+")
                    f.write(str(metrics) + "\n")

                    for m in ["training_metrics", "cross_validation_metrics", "test_metrics"]:
                        if m in candidate.data and candidate.data[m]:
                            f.write(m + ' ' +  str(candidate.data[m][0]['value']) + "\n")
                    # f.write(str(candidate.data['training_metrics'][0]['value']) + "\n")
                    # f.write(str(candidate.data['cross_validation_metrics'][0]['value']) + "\n")
                    # f.write(str(candidate.data['test_metrics'][0]['value']) + "\n")
                    f.close()
                except:
                    self._logger.exception('[ERROR] Save training results Failed!')
                    raise NotSupportedError(
                        '[ERROR] Save training results Failed!')

        # shutdown the cache manager
        manager.shutdown()


    def update_UCT_score(self, index: int, report: typing.Dict):
        self.update_history(index, report)

        alpha = 0.01
        self.normalize = self.exec_history[['reward', 'exe_time', 'trial']]
        scale = (self.normalize.max() - self.normalize.min())
        scale.replace(to_replace=0, value=1, inplace=True)
        self.normalize = (self.normalize - self.normalize.min()) / scale
        self.normalize.clip(lower=0.01, upper=1, inplace=True)

        for i in range(len(self.uct_score)):
            self.uct_score[i] = self.compute_UCT(i)

        self._logger.info(STYLE+"[INFO] UCT updated: %s", self.uct_score)

    def update_history(self, index, report):
        self.total_run += report['sim_count']
        self.total_time += report['time']
        row = self.exec_history.iloc[index]
        update = {
            'trial': row['trial'] + report['sim_count'],
            'exe_time': row['exe_time'] + report['time'],
            'candidate': report['candidate'],
        }
        if report['reward'] is not None:
            update['reward'] = (
                    (row['reward'] * row['trial'] + report['reward'] * report['sim_count']) /
                    (row['trial'] + report['sim_count'])
            )
            update['best_value'] = max(report['reward'], row['best_value'])

        for k in update:
            self.exec_history.iloc[index][k] = update[k]


    def write_training_results(self):
        # load trained pipelines
        self._logger.info("[WARN] write_training_results")

        if len(self.exec_history) == 0:
            return None

        # if best_info and best_info['best_val']:
        best_template, best_report = max(self.exec_history.iterrows(),
                                         key=lambda r: r[1]['best_value'])

        if best_template:
            self._logger.info("[INFO] Best template name:{}".format(best_template))
            self._logger.info("[INFO] Best value:{}".format(best_report['best_value']))
            self._logger.info("[INFO] Best Candidate:{}".format(
                pprint.pformat(best_report['candidate'])))
        return None
