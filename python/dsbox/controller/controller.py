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

from multiprocessing import Pool, current_process, Manager
from math import sqrt, log
import traceback
import importlib
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

import numpy as np
import pandas as pd
import d3m
import dsbox.template.runtime as runtime

from d3m.metadata.problem import TaskType
from d3m.container.dataset import Dataset
from d3m.container.dataset import D3MDatasetLoader
from d3m.exceptions import NotSupportedError
from d3m.exceptions import InvalidArgumentValueError
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.base import Metadata
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

from dsbox.combinatorial_search.search_utils import get_target_columns
from dsbox.combinatorial_search.search_utils import random_choices_without_replacement
from dsbox.template.template import DSBoxTemplate

__all__ = ['Status', 'Controller']

import copy
import pprint
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# FIXME: we only need this for testing
import pandas as pd

FILE_FORMATTER = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
FILE_LOGGING_LEVEL = logging.WARNING
LOG_FILENAME = 'dsbox.log'
CONSOLE_LOGGING_LEVEL = logging.INFO
# CONSOLE_LOGGING_LEVEL = logging.DEBUG
CONSOLE_FORMATTER = "[%(levelname)s] - %(name)s - %(message)s"


def auto_regress_convert(dataset: "Dataset", problem: "Metadata"):
    '''
    do auto convert for timeseriesforecasting prob
    '''
    problem = problem.query(())
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


def remove_empty_targets(dataset: "Dataset", problem: "Metadata"):
    '''
    will automatically remove empty targets in training
    '''
    problem = problem.query(())
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


def split_dataset(dataset, problem_info: typing.Dict, problem_loc=None, *, random_state=42, test_size=0.2, n_splits=1):
    '''
        Split dataset into training and test
    '''

    # hard coded unsplit dataset type
    # TODO: check whether "speech" type should be put into this list or not
    data_type_cannot_split = ["graph","edgeList", "audio"]
    task_type_can_split = ["CLASSIFICATION","REGRESSION"]

    task_type = problem_info["task_type"]#['problem']['task_type'].name  # 'classification' 'regression'
    res_id = problem_info["res_id"]
    target_index = problem_info["target_index"]
    data_type = problem_info["data_type"]

    def generate_split_data(dataset, res_id):
        train = dataset[res_id].iloc[train_index, :]
        test = dataset[res_id].iloc[test_index, :]

        # Generate training dataset
        train_dataset = copy.copy(dataset)
        dataset_metadata = dict(train_dataset.metadata.query(()))
        dataset_metadata['id'] = dataset_metadata['id'] + '_' + str(uuid.uuid4())
        train_dataset.metadata = train_dataset.metadata.update((), dataset_metadata)

        train_dataset[res_id] = train
        meta = dict(train_dataset.metadata.query((res_id,)))
        dimension = dict(meta['dimension'])
        meta['dimension'] = dimension
        dimension['length'] = train.shape[0]
        # print(meta)
        train_dataset.metadata = train_dataset.metadata.update((res_id,), meta)
        # pprint(dict(train_dataset.metadata.query((res_id,))))

        # Generate testing dataset
        test_dataset = copy.copy(dataset)
        test_dataset[res_id] = test
        dataset_metadata = dict(test_dataset.metadata.query(()))
        dataset_metadata['id'] = dataset_metadata['id'] + '_' + str(uuid.uuid4())
        test_dataset.metadata = dataset.metadata.update((), dataset_metadata)

        meta = dict(test_dataset.metadata.query((res_id,)))
        dimension = dict(meta['dimension'])
        meta['dimension'] = dimension
        dimension['length'] = test.shape[0]
        # print(meta)
        test_dataset.metadata = test_dataset.metadata.update((res_id,), meta)
        # pprint(dict(test_dataset.metadata.query((res_id,))))
        return (train_dataset, test_dataset)

    train_return = []
    test_return = []

    cannot_split = False

    for each in data_type:
        if each in data_type_cannot_split:
            cannot_split = True
            break

    # check second time if the program think we still can split
    if not cannot_split:
        if task_type is not list:
            task_type_check = [task_type]

        for each in task_type_check:
            if each not in task_type_can_split:
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
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])
                for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):
                    train_dataset,test_dataset = generate_split_data(dataset,res_id)
                    train_return.append(train_dataset)
                    test_return.append(test_dataset)
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
                train_dataset,test_dataset = generate_split_data(dataset,res_id)
                train_return.append(train_dataset)
                test_return.append(test_dataset)

    return (train_return, test_return)

class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148


class Controller:
    TIMEOUT = 59  # in minutes

    def __init__(self, development_mode: bool = False, run_single_template_name: str = "") -> None:
        self.development_mode: bool = development_mode

        self.run_single_template_name = run_single_template_name

        self.config: typing.Dict = {}

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

        self._did_we_post_process: bool = False

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

        # self.dataset, self.test_dataset = split_dataset(self.all_dataset, self.problem, config['problem_schema'])
        # self.dataset = runtime.add_target_columns_metadata(self.dataset, self.problem_doc_metadata)
        # self.test_dataset = runtime.add_target_columns_metadata(self.test_dataset, self.problem_doc_metadata)
        # self.test_dataset = runtime.add_target_columns_metadata(self.test_dataset, self.problem_doc_metadata)

        # Templates
        self.load_templates()

    def _load_schema(self, config):
        # config
        self.config = config

        # Problem
        self.problem = parse_problem_description(config['problem_schema'])
        self.problem_doc_metadata = runtime.load_problem_doc(os.path.abspath(config['problem_schema']))

        # Dataset
        self.dataset_schema_file = config['dataset_schema']

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
        self.problem_info["target_index"] = self.problem['inputs'][i]['targets'][0]['column_index']

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

    def load_templates(self) -> None:
        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']
        # find the data resources type
        self.taskSourceType = set()  # set the type to be set so that we can ignore the repeat elements
        with open(self.dataset_schema_file, 'r') as dataset_description_file:
            dataset_description = json.load(dataset_description_file)
            for each_type in dataset_description["dataResources"]:
                self.taskSourceType.add(each_type["resType"])

        self.template = self.template_library.get_templates(self.task_type, self.task_subtype, self.taskSourceType)
        self.problem_info["data_type"] = self.taskSourceType
        # find the maximum dataset split requirements
        for each_template in self.template:
            for each_step in each_template.template['steps']:
                if "runtime" in each_step and "test_validation" in each_step["runtime"]:
                    split_times = int(each_step["runtime"]["test_validation"])
                    if split_times > self.max_split_times:
                        self.max_split_times = split_times

    def _process_pipeline_submission(self) -> None:
        pipelines_root: str = os.path.join(self.output_directory, 'pipelines')
        executables_root: str = os.path.join(self.output_directory, 'executables')
        supporting_root: str = os.path.join(self.output_directory, 'supporting_files')

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

    def write_training_results(self):
        # load trained pipelines
        if not self._did_we_post_process:
            self._logger.warning("write_training_results")
            self._did_we_post_process = True
            self._process_pipeline_submission()

        return None

    def _log_search_results(self, report: typing.Dict[str, typing.Any]):
        candidate = report['configuration']

        print("-" * 20)
        print("[INFO] Final Search Results:")
        pprint.pprint(candidate)

        if candidate is None:
            self._logger.error("[ERROR] No candidate found during search!")
            print("[INFO] cross_validation_metrics:", None)
        else:
            value = report['cross_validation_metrics'][0]['value']
            print("[INFO] cross_validation_metrics:", value)

            self._logger.info("******************\n[INFO] Writing results")
            self._logger.info(str(report) + " " + str(value))
            if report['training_metrics']:
                self._logger.info('Training {} = {}'.format(
                    report['training_metrics'][0]['metric'],
                    report['training_metrics'][0]['value']))
            if report['cross_validation_metrics']:
                self._logger.info('CV {} = {}'.format(
                    report['cross_validation_metrics'][0]['metric'],
                    report['cross_validation_metrics'][0]['value']))
            if report['test_metrics']:
                self._logger.info('Validation {} = {}'.format(
                    report['test_metrics'][0]['metric'],
                    report['test_metrics'][0]['value']))

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
        pid: int = os.fork()
        if pid == 0:  # run the search in the child process
            # self._run_SerialBaseSearch()
            self._run_ParallelBaseSearch()
            # self._run_RandomDimSearch()

            print("[INFO] End of Search")
            os._exit(0)
        else:
            status = os.wait()
            print("[INFO] Search Status:")
            pprint.pprint(status)
        print("END OF FORK")

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
        report = searchMethod.search(num_iter=40)

        self._log_search_results(report=report)

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

    def generate_dataset_splits(self):
        # For now just use the first template
        self.all_dataset = remove_empty_targets(self.all_dataset, self.problem_doc_metadata)
        self.all_dataset = auto_regress_convert(self.all_dataset, self.problem_doc_metadata)
        runtime.add_target_columns_metadata(self.all_dataset, self.problem_doc_metadata)
        # split the dataset first time
        self.train_dataset1, self.test_dataset1 = split_dataset(dataset=self.all_dataset,
                                                                problem_info=self.problem_info,
                                                                problem_loc=self.config[
                                                                    'problem_schema'])
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
            self.train_dataset2, self.test_dataset2 = split_dataset(dataset=self.train_dataset1,
                                                                    problem_info=self.problem_info,
                                                                    problem_loc=self.config[
                                                                        'problem_schema'],
                                                                    test_size=0.1,
                                                                    n_splits=self.max_split_times)
            if len(self.train_dataset2) < 1:
                self._logger.error(
                    "Some error happend with train_dataset1 split: The length of splitted dataset is less than 1")
            if len(self.test_dataset2) < 1:
                self._logger.error("Split failed on train_dataset1.")
                self.test_dataset2 = None
        else:
            self.train_dataset2 = None
            self.test_dataset2 = None

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

        pipeline_load, run = FittedPipeline.load(folder_loc=self.output_directory,
                                                 pipeline_id=fitted_pipeline_id,
                                                 log_dir=self.output_logs_dir)

        print("[INFO] Pipeline load finished")

        print("[INFO] testing data:")
        # pprint(self.test_dataset.head())

        # pipeline_load.runtime.produce(inputs=[self.test_dataset])
        self.all_dataset = auto_regress_convert(self.all_dataset, self.problem_doc_metadata)
        run.produce(inputs=[self.all_dataset])
        try:
            step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        except:
            print("Warning: searching the output step number failed! Will use the last step's output of the pipeline.")
            # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            step_number_output = len(run.produce_outputs) - 1

        # get the target column name
        try:
            with open(self.dataset_schema_file, 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_resource in dataset_description["dataResources"]:
                    if "columns" in each_resource:
                        for each_column in each_resource["columns"]:
                            if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
                                prediction_class_name = each_column["colName"]
        except:
            print("[Warning] Can't find the prediction class name, will use default name.")
            prediction_class_name = "prediction"

        prediction = run.produce_outputs[step_number_output]

        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            prediction_col_name = prediction.columns[0]
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[['d3mIndex', prediction_col_name]]
            prediction = prediction.rename(columns={prediction_col_name: prediction_class_name})
        prediction_folder_loc = self.output_directory + "/predictions/" + fitted_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)
        print("[INFO] Finished: prediction results saving finished")
        print("[INFO] The prediction results is stored at: ", prediction_folder_loc)
        return Status.OK

    def test(self) -> Status:
        """
        First read the fitted pipeline and then run trained pipeline on test data.
        """
        self._logger.info("[INFO] Start test function")
        outputs_loc, pipeline_load, read_pipeline_id, run = \
            self.load_pipe_runtime()

        self._logger.info("[INFO] Pipeline load finished")

        self._logger.info("[INFO] testing data")
        self.all_dataset = auto_regress_convert(self.all_dataset, self.problem_doc_metadata)
        run.produce(inputs=[self.all_dataset])
        try:
            step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        except:
            self._logger.error("Warning: searching the output step number failed! "
                               "Will use the last step's output of the pipeline.")
            # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            step_number_output = len(run.produce_outputs) - 1

        # get the target column name
        try:
            with open(self.dataset_schema_file, 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_resource in dataset_description["dataResources"]:
                    if "columns" in each_resource:
                        for each_column in each_resource["columns"]:
                            if "suggestedTarget" in each_column["role"] or "target" in each_column["role"]:
                                prediction_class_name = each_column["colName"]
        except:
            self._logger.error("[Warning] Can't find the prediction class name, will use default name 'prediction'.")
            prediction_class_name = "prediction"

        prediction = run.produce_outputs[step_number_output]

        # if the prediction results do not have d3m_index column
        if 'd3mIndex' not in prediction.columns:
            d3m_index = get_target_columns(self.all_dataset, self.problem_doc_metadata)["d3mIndex"]
            d3m_index = d3m_index.reset_index().drop(columns=['index'])
            prediction_col_name = prediction.columns[0]
            prediction['d3mIndex'] = d3m_index
            prediction = prediction[['d3mIndex', prediction_col_name]]
            prediction = prediction.rename(columns={prediction_col_name: prediction_class_name})
        prediction_folder_loc = outputs_loc + "/predictions/" + read_pipeline_id
        folder = os.path.exists(prediction_folder_loc)
        if not folder:
            os.makedirs(prediction_folder_loc)
        prediction.to_csv(prediction_folder_loc + "/predictions.csv", index=False)
        self._logger.info("[INFO] Finished: prediction results saving finished")
        self._logger.info("[INFO] The prediction results is stored at: {}".format(prediction_folder_loc))
        return Status.OK

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
