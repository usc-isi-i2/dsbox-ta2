import enum
import json
import logging
import os
import random
import typing

import d3m
import dsbox.template.runtime as runtime

from d3m.container.dataset import Dataset
from d3m.container.dataset import D3MDatasetLoader
from d3m.exceptions import NotSupportedError
from d3m.exceptions import InvalidArgumentValueError
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.base import Metadata
from d3m.metadata.problem import TaskType
from d3m.metadata.problem import TaskSubtype
from d3m.metadata.problem import parse_problem_description

from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.template.library import TemplateDescription
from dsbox.template.library import TemplateLibrary
from dsbox.template.library import SemanticTypeDict
from dsbox.template.search import ConfigurationSpace
from dsbox.template.search import SimpleConfigurationSpace
from dsbox.template.search import TemplateDimensionalSearch
from dsbox.template.search import get_target_columns
from dsbox.template.template import DSBoxTemplate

__all__ = ['Status', 'Controller']

import copy
import pprint
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

# FIXME: we only need this for testing
import pandas as pd

FORMATTER = "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s"
LOGGING_LEVER = logging.DEBUG
LOG_FILENAME = 'dsbox.log'

def split_dataset(dataset, problem, problem_loc=None, *, random_state=42, test_size=0.2):
    '''
    Split dataset into training and test
    '''

    task_type : TaskType = problem['problem']['task_type']  # 'classification' 'regression'

    for i in range(len(problem['inputs'])):
        if 'targets' in problem['inputs'][i]:
            break

    res_id = problem['inputs'][i]['targets'][0]['resource_id']
    target_index = problem['inputs'][i]['targets'][0]['column_index']

    try:
        splits_file = problem_loc.rsplit("/", 1)[0] + "/dataSplits.csv"

        df = pd.read_csv(splits_file)

        train_test = df[df.columns[1]]
        train_indices = df[train_test == 'TRAIN'][df.columns[0]]
        test_indices = df[train_test == 'TEST'][df.columns[0]]

        train = dataset[res_id].iloc[train_indices]
        test = dataset[res_id].iloc[test_indices]

        use_test_splits = False

        print("[INFO] Succesfully parsed test data")
    except:
        if task_type == TaskType.CLASSIFICATION:
            # Use stratified sample to split the dataset
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])
            for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):
                train = dataset[res_id].iloc[train_index,:]
                test = dataset[res_id].iloc[test_index,:]
        else:
            # Use random split
            if not task_type == TaskType.REGRESSION:
                print('USING Random Split to split task type: {}'.format(task_type))
            ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            ss.get_n_splits(dataset[res_id])
            for train_index, test_index in ss.split(dataset[res_id]):
                train = dataset[res_id].iloc[train_index,:]
                test = dataset[res_id].iloc[test_index,:]

        print("[INFO] Failed test data parse/ using stratified kfold data instead")

    # Generate training dataset
    train_dataset = copy.copy(dataset)
    train_dataset[res_id] = train
    meta = dict(train_dataset.metadata.query((res_id,)))
    dimension = dict(meta['dimension'])
    meta['dimension'] = dimension
    dimension['length'] = train.shape[0]
    print(meta)
    train_dataset.metadata = train_dataset.metadata.update((res_id,), meta)
    pprint.pprint(dict(train_dataset.metadata.query((res_id,))))

    # Generate testing dataset
    test_dataset = copy.copy(dataset)
    test_dataset[res_id] = test
    meta = dict(test_dataset.metadata.query((res_id,)))
    dimension = dict(meta['dimension'])
    meta['dimension'] = dimension
    dimension['length'] = test.shape[0]
    print(meta)
    test_dataset.metadata = test_dataset.metadata.update((res_id,), meta)
    pprint.pprint(dict(test_dataset.metadata.query((res_id,))))


    return (train_dataset, test_dataset)



class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148


class Controller:
    TIMEOUT = 59  # in minutes

    def __init__(self, development_mode: bool = False, run_single_template: str = "") -> None:
        self.development_mode: bool = development_mode

        self.run_single_template = run_single_template

        # self.config: typing.Dict = {}

        # Problem
        self.problem: typing.Dict = {}
        self.task_type: TaskType = None
        self.task_subtype: TaskSubtype = None
        self.problem_doc_metadata: Metadata = None

        # Dataset
        self.dataset_schema_file: str = ""
        self.dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.taskSourceType: typing.Set[str]  = set()  # str from SEMANTIC_TYPES

        # Resource limits
        self.num_cpus: int = 0
        self.ram: int = 0  # concurrently ignored
        self.timeout: int = 0  # in seconds

        # Templates
        if self.run_single_template:
            self.template_library = TemplateLibrary(run_single_template=run_single_template)
        else:
            self.template_library = TemplateLibrary()
        self.template: typing.List[DSBoxTemplate] = []

        # Primitives
        self.primitive: typing.Dict = d3m.index.search()

        # set random seed
        random.seed(4676)

        # Output directories
        self.output_directory: str = '/output/'
        self.output_pipelines_dir: str = ""
        self.output_executables_dir: str = ""
        self.output_supporting_files_dir: str = ""
        self.output_temp_dir: str = ""
        self.output_logs_dir: str = ""

        self._logger = None

    def initialize_from_config(self, config: typing.Dict) -> None:

        self._load_schema(config)
        self._create_output_directory(config)

        # Dataset
        loader = D3MDatasetLoader()

        #dataset_uri = 'file://{}'.format(os.path.abspath(self.dataset_schema_file))
        #self.dataset = loader.load(dataset_uri=dataset_uri)

        # train dataset
        train_dataset_uri = 'file://{}'.format(os.path.abspath(config['train_data_schema']))
        self.dataset = loader.load(dataset_uri=train_dataset_uri)

        # test dataset
        test_dataset_uri = 'file://{}'.format(os.path.abspath(config['test_data_schema']))
        self.test_dataset = loader.load(dataset_uri=test_dataset_uri)

        # Templates
        self.load_templates()

    def initialize_from_config_train_test(self, config: typing.Dict) -> None:

        self._load_schema(config)
        self._create_output_directory(config)

        # Dataset
        loader = D3MDatasetLoader()

        json_file = os.path.abspath(self.dataset_schema_file)
        all_dataset_uri = 'file://{}'.format(json_file)
        self.all_dataset = loader.load(dataset_uri=all_dataset_uri)

        self.dataset, self.test_dataset = split_dataset(self.all_dataset, self.problem, config['problem_schema'])

        self.test_dataset = runtime.add_target_columns_metadata(self.test_dataset, self.problem_doc_metadata)

        # Templates
        self.load_templates()

    def _load_schema(self, config):
        # Problem
        self.problem = parse_problem_description(config['problem_schema'])
        self.problem_doc_metadata = runtime.load_problem_doc(os.path.abspath(config['problem_schema']))

        # Dataset
        self.dataset_schema_file = config['dataset_schema']

        # Resource limits
        self.num_cpus = int(config.get('cpus', 0))
        self.ram = config.get('ram', 0)
        self.timeout = (config.get('timeout', self.TIMEOUT)) * 60
        try:
            self.saved_pipeline_id = config['saved_pipeline_ID']
        except:
            self.saved_pipeline_id = ""

    def _create_output_directory(self, config):
        '''
        Create output sub-directories based on Summer 2018 evaluation layout.

        For the Summer 2018 evaluation the top-level output dir is '/output'
        '''
        if 'saving_folder_loc' in config:
            self.output_directory = os.path.abspath(config['saving_folder_loc'])

        if 'pipelines_root' in config:
            self.output_pipelines_dir = os.path.abspath(config['pipelines_root'])
        else:
            self.output_pipelines_dir = os.path.join(self.output_directory, 'pipelines')

        if 'executables_root' in config:
            self.output_executables_dir = os.path.abspath(config['executables_root'])
        else:
            self.output_executables_dir = os.path.join(self.output_directory, 'executables')

        if 'supporting_files_root' in config:
            self.output_supporting_files_dir = os.path.abspath(config['supporting_files_root'])
        else:
            self.output_supporting_files_dir = os.path.join(self.output_directory, 'supporting_files')

        if 'temp_storage_root' in config:
            self.output_temp_dir = os.path.abspath(config['temp_storage_root'])
        else:
            self.output_temp_dir = os.path.join(self.output_directory, 'temp')

        if 'pipeline_logs_root' in config:
            self.output_logs_dir = os.path.abspath(config['pipeline_logs_root'])
        else:
            self.output_logs_dir = os.path.join(self.output_directory, 'logs')

        # Make directories if they do not exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        for path in [self.output_pipelines_dir, self.output_executables_dir, self.output_supporting_files_dir,
                     self.output_temp_dir, self.output_logs_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        self._log_init()
        self._logger.info('Top level output directory: %s' % self.output_directory)

    def _log_init(self) -> None:
        logging.basicConfig(
            level=LOGGING_LEVER,
            format=FORMATTER,
            datefmt='%m-%d %H:%M',
            filename=os.path.join(self.output_logs_dir, LOG_FILENAME),
            filemode='w'
        )

        self._logger = logging.getLogger(__name__)

        if self._logger.getEffectiveLevel() <= 10:
            os.makedirs(os.path.join(self.output_logs_dir, "dfs"), exist_ok=True)

        # ch = logging.StreamHandler()
        # ch.setFormatter(logging.Formatter(FORMATTER))
        # ch.setLevel(logging.INFO)
        # self._logger.addHandler(ch)


    def load_templates(self) -> None:
        self.task_type = self.problem['problem']['task_type']
        self.task_subtype = self.problem['problem']['task_subtype']
        # find the data resources type
        self.taskSourceType = set() # set the type to be set so that we can ignore the repeat elements
        with open(self.dataset_schema_file,'r') as dataset_description_file:
            dataset_description = json.load(dataset_description_file)
            for each_type in dataset_description["dataResources"]:
                self.taskSourceType.add(each_type["resType"])
        self.template = self.template_library.get_templates(self.task_type, self.task_subtype, self.taskSourceType)

    def write_training_results(self):
        # load trained pipelines
        print("[WARN] write_training_results")

        return None

        d = self.output_pipelines_dir
        # for now, the program will automatically load the newest created file in the folder
        files = [os.path.join(d, f) for f in os.listdir(d)]
        exec_pipelines = []
        for f in files:
            fname = f.split('/')[-1].split('.')[0]
            pipeline_load = FittedPipeline.load(folder_loc=self.output_executables_dir,
                                                pipeline_id=fname,
                                                dataset=self.dataset,
                                                log_dir=self.output_logs_dir)
            exec_pipelines.append(pipeline_load)


        print("[INFO] wtr:",exec_pipelines)
        # sort the pipelins
        # TODO add the sorter method
        # self.exec_pipelines = self.get_pipeline_sorter().sort_pipelines(self.exec_pipelines)

        # write the results
        pipelinesfile = open("somefile.ext",'W+') # TODO check this address
        print("Found total %d successfully executing pipeline(s)..." % len(exec_pipelines))
        pipelinesfile.write("# Pipelines ranked by (adjusted) metrics (%s)\n" % self.problem.metrics)
        for pipe in exec_pipelines:
            metric_values = []
            for metric in pipe.planner_result.metric_values.keys():
                metric_value = pipe.planner_result.metric_values[metric]
                metric_values.append("%s = %2.4f" % (metric, metric_value))

            pipelinesfile.write("%s ( %s ) : %s\n" % (pipe.id, pipe, metric_values))

        pipelinesfile.flush()
        pipelinesfile.close()


    def train(self) -> Status:
        """
        Generate and train pipelines.
        """
        if not self.template:
            return Status.PROBLEM_NOT_IMPLEMENT

        #self._check_and_set_dataset_metadata()

        # For now just use the first template
        # TODO: sample based on DSBoxTemplate.importance()
        template = self.template[0]

        space = template.generate_configuration_space()

        metrics = self.problem['problem']['performance_metrics']

        # search = TemplateDimensionalSearch(template, space, d3m.index.search(), self.dataset, self.dataset, metrics)
        if self.test_dataset is None:
            search = TemplateDimensionalSearch(
                template, space, d3m.index.search(), self.problem_doc_metadata, self.dataset,
                self.dataset, metrics, output_directory=self.output_directory, log_dir=self.output_logs_dir, num_workers=self.num_cpus)
        else:
            search = TemplateDimensionalSearch(
                template, space, d3m.index.search(), self.problem_doc_metadata, self.dataset,
                self.test_dataset, metrics, output_directory=self.output_directory, log_dir=self.output_logs_dir, num_workers=self.num_cpus)

        candidate, value = search.search_one_iter()

        # assert "fitted_pipe" in candidate, "argument error!"

        if candidate is None:
            print("[ERROR] not candidate!")
            return Status.PROBLEM_NOT_IMPLEMENT
        else:
            print("******************\n[INFO] Writing results")
            print(candidate.data)
            print(candidate, value)
            if candidate.data['training_metrics']:
                print('Training {} = {}'.format(
                    candidate.data['training_metrics'][0]['metric'],
                    candidate.data['training_metrics'][0]['value']))
            if candidate.data['cross_validation_metrics']:
                print('Training {} = {}'.format(
                    candidate.data['cross_validation_metrics'][0]['metric'],
                    candidate.data['cross_validation_metrics'][0]['value']))
            if candidate.data['test_metrics']:
                print('Test {} = {}'.format(
                    candidate.data['test_metrics'][0]['metric'],
                    candidate.data['test_metrics'][0]['value']))

            # FIXME: code used for doing experiments, want to make optionals
            # pipeline = FittedPipeline.create(configuration=candidate,
            #                             dataset=self.dataset)

            dataset_name = self.output_executables_dir.rsplit("/", 2)[1]
            save_location = os.path.join(self.output_logs_dir, dataset_name + ".txt")

            print("******************\n[INFO] Saving training results in", save_location)
            try:
                f = open(save_location, "w+")
                f.write(str(metrics) + "\n")
                f.write(str(candidate.data['training_metrics'][0]['value']) + "\n")
                f.write(str(candidate.data['test_metrics'][0]['value']) + "\n")
                f.close()
            except:
                raise NotSupportedError(
                    '[ERROR] Save training results Failed!')

            print("******************\n[INFO] Saving Best Pipeline")
            # save the pipeline

            try:
                # pipeline = FittedPipeline.create(configuration=candidate,
                #                                  dataset=self.dataset)
                fitted_pipeline = candidate.data['fitted_pipeline']
                fitted_pipeline.save(self.output_directory)
            except:
                raise NotSupportedError(
                    '[ERROR] Save Failed!')
            ####################
            return Status.OK

    def test(self) -> Status:
        """
        First read the fitted pipeline and then run trained pipeline on test data.
        """
        print("[INFO] Start test function")
        outputs_loc, pipeline_load, read_pipeline_id, run = \
            self.load_pipe_runtime()

        print("[INFO] Pipeline load finished")

        print("[INFO] testing data:")
        # pprint(self.test_dataset.head())
        # pipeline_load.runtime.produce(inputs=[self.test_dataset])
        run.produce(inputs=[self.test_dataset])
        try:
            step_number_output = int(pipeline_load.pipeline.outputs[0]['data'].split('.')[1])
        except:
            print("Warning: searching the output step number failed! Will use the last step's output of the pipeline.")
            # step_number_output = len(pipeline_load.runtime.produce_outputs) - 1
            step_number_output = len(run.produce_outputs) - 1

        # get the target column name
        try:
            with open(self.dataset_schema_file,'r') as dataset_description_file:
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
            d3m_index = get_target_columns(self.test_dataset, self.problem_doc_metadata)["d3mIndex"]
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
        print("[INFO] Finished: prediction results saving finished")
        print("[INFO] The prediction results is stored at: ", prediction_folder_loc)
        return Status.OK

    def load_pipe_runtime(self):
        d = os.path.expanduser(self.output_directory + '/pipelines')
        read_pipeline_id = self.saved_pipeline_id
        if read_pipeline_id == "":
            print(
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

    def generate_configuration_space(self, template_desc: TemplateDescription, problem: typing.Dict, dataset: typing.Optional[Dataset]) -> ConfigurationSpace:
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
        print(values)
        # values: typing.Dict[DimensionName, typing.List] = {}
        return SimpleConfigurationSpace(values)

    def _check_and_set_dataset_metadata(self) -> None:
        # Need to make sure the Target and TrueTarget column semantic types are set
        if self.task_type == TaskType.CLASSIFICATION or self.task_type == TaskType.REGRESSION:

            # start from last column, since typically target is the last column
            for index in range(self.dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length']-1, -1, -1):
                column_semantic_types = self.dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if ('https://metadata.datadrivendiscovery.org/types/Target' in column_semantic_types
                        and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in column_semantic_types):
                    return

            # If not set, use sugested target column
            for index in range(self.dataset.metadata.query(('0', ALL_ELEMENTS))['dimension']['length']-1, -1, -1):
                column_semantic_types = self.dataset.metadata.query(
                    ('0', ALL_ELEMENTS, index))['semantic_types']
                if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in column_semantic_types:
                    column_semantic_types = list(column_semantic_types) + ['https://metadata.datadrivendiscovery.org/types/Target',
                                                                           'https://metadata.datadrivendiscovery.org/types/TrueTarget']
                    self.dataset.metadata = self.dataset.metadata.update(
                        ('0', ALL_ELEMENTS, index), {'semantic_types': column_semantic_types})
                    return

            raise InvalidArgumentValueError(
                'At least one column should have semantic type SuggestedTarget')
