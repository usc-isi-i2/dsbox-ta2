import enum
import os
import random
import typing

import d3m

from d3m.container.dataset import Dataset, D3MDatasetLoader
from d3m.metadata.base import ALL_ELEMENTS
from d3m.metadata.problem import parse_problem_description, TaskType, TaskSubtype
from d3m.exceptions import NotSupportedError, InvalidArgumentValueError

from dsbox.template.library import TemplateLibrary, TemplateDescription, SemanticTypeDict
from dsbox.template.search import TemplateDimensionalRandomHyperparameterSearch, TemplateDimensionalSearch, ConfigurationSpace, SimpleConfigurationSpace, PythonPath, DimensionName
from dsbox.template.template import TemplatePipeline,to_digraph
from dsbox.pipeline.fitted_pipeline import FittedPipeline

__all__ = ['Status', 'Controller']


class Status(enum.Enum):
    OK = 0
    PROBLEM_NOT_IMPLEMENT = 148


class Controller:
    TIMEOUT = 59  # in minutes

    def __init__(self, library_dir: str, development_mode: bool = False) -> None:
        self.library_dir: str = os.path.abspath(library_dir)
        self.development_mode: bool = development_mode

        self.config: typing.Dict = {}

        # Problem
        self.problem: typing.Dict = {}
        self.task_type: TaskType = None
        self.task_subtype: TaskSubtype = None

        # Dataset
        self.dataset: Dataset = None

        # Resource limits
        self.num_cpus: int = 0
        self.ram: int = 0  # concurrently ignored
        self.timeout: int = 0  # in seconds

        # Templates
        self.template_library = TemplateLibrary()
        self.template_description: typing.List[TemplateDescription] = []

        # Primitives
        self.primitive: typing.Dict = d3m.index.search()

        # set random seed
        random.seed(4676)

    def initialize_from_config(self, config: typing.Dict) -> None:
        self.config = config

        # Problem
        self.problem = parse_problem_description(config['problem_schema'])

        # Dataset
        loader = D3MDatasetLoader()
        dataset_uri = 'file://{}'.format(
            os.path.abspath(config['dataset_schema']))
        self.dataset = loader.load(dataset_uri=dataset_uri)

        # Resource limits
        self.num_cpus = int(config.get('cpus', 0))
        self.ram = config.get('ram', 0)
        self.timeout = (config.get('timeout', self.TIMEOUT)) * 60

        # Templates
        self.load_templates(
            self.problem['problem']['task_type'], self.problem['problem']['task_subtype'])

    def load_templates(self, task_type: TaskType, task_subtype: TaskSubtype) -> None:
        self.task_type = task_type
        self.task_subtype = task_subtype

        self.template_description = self.template_library.get_templates(
            self.task_type, self.task_subtype)

    def write_training_results(self):
        # load trained pipelines
        d = os.path.expanduser(self.config['executables_root'] + '/pipelines')
        # for now, the program will automatically load the newest created file in the folder
        files = [os.path.join(d, f) for f in os.listdir(d)]
        exec_pipelines = []
        for f in files:
            fname = f.split('/')[-1].split('.')[0]
            pipeline_load = FittedPipeline.load(folder_loc=self.config['executables_root'],
                                                pipeline_id=fname,
                                                dataset=self.dataset)
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

            pipelinesfile.write("%s ( %s ) : %s\n" % (pipeline.id, pipe, metric_values))

        pipelinesfile.flush()
        pipelinesfile.close()


    def train(self) -> Status:
        """
        Generate and train pipelines.
        """
        if not self.template_description:
            return Status.PROBLEM_NOT_IMPLEMENT

        self._check_and_set_dataset_metadata()

        # For now just use the first template
        template = self.template_description[0]

        space = self.generate_configuration_space(
            template, self.problem, self.dataset)

        metrics = self.problem['problem']['performance_metrics']

        # search = TemplateDimensionalSearch(template, space, d3m.index.search(), self.dataset, self.dataset, metrics)
        search = TemplateDimensionalSearch(template, space, d3m.index.search(), self.dataset, self.dataset, metrics)
        candidate, value = search.search_one_iter()
        if candidate is None:
            return Status.PROBLEM_NOT_IMPLEMENT
        else:
            print("????")
            print(candidate.data)
            print(candidate, value)
            print("###", value)

            # save the pipeline
            pipeline = FittedPipeline.create(configuration = candidate, dataset = self.dataset)
            pipeline.save(self.config['executables_root'])

            return Status.OK

    def test(self) -> Status:
        """
        First read the fitted pipeline and then run trained pipeline on test data.
        """
        print("=====~~~~~~~~~~~  new pipeline loading function test  ~~~~~~~~~~~=====")


        d = os.path.expanduser(self.config['executables_root'] + '/pipelines') 
        # for now, the program will automatically load the newest created file in the folder
        files = [os.path.join(d, f) for f in os.listdir(d)]
        files.sort(key=lambda f: os.stat(f).st_mtime)
        lastmodified = files[-1]
        read_pipeline_id = lastmodified.split('/')[-1].split('.')[0]
        
        pipeline_load = FittedPipeline.load(folder_loc = self.config['executables_root'], pipeline_id = read_pipeline_id, dataset = self.dataset)

        print("=====~~~~~~~~~~~  new pipeline loading function finished  ~~~~~~~~~~~=====")
        
        return Status.OK

    # def generate_configuration_space(self, template_desc: TemplateDescription, problem: typing.Dict, dataset: typing.Optional[Dataset]) -> ConfigurationSpace:
    #     """
    #     Generate search space
    #     """

    #     # TODO: Need to update dsbox.planner.common.ontology.D3MOntology and dsbox.planner.common.ontology.D3MPrimitiveLibrary, and integrate with them

    #     values: typing.Dict[DimensionName, typing.List] = {}
    #     for name, step in template_desc.template.template_nodes.items():
    #         if step.semantic_type == SemanticType.CLASSIFIER:
    #             values[DimensionName(name)] = ['d3m.primitives.common_primitives.RandomForestClassifier', 'd3m.primitives.sklearn_wrap.SKSGDClassifier']
    #         elif step.semantic_type == SemanticType.REGRESSOR:
    #             values[DimensionName(name)] = ['d3m.primitives.common_primitives.LinearRegression',
    #                                            'd3m.primitives.sklearn_wrap.SKSGDRegressor',
    #                                            'd3m.primitives.sklearn_wrap.SKRandomForestRegressor']
    #         elif step.semantic_type == SemanticType.ENCODER:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #         elif step.semantic_type == SemanticType.IMPUTER:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #         elif step.semantic_type == SemanticType.FEATURER_GENERATOR:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #         elif step.semantic_type == SemanticType.FEATURER_SELECTOR:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #         elif step.semantic_type == SemanticType.UNDEFINED:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #         else:
    #             raise NotSupportedError('Template semantic type not supported: {}'.format(step.semantic_type))
    #     return SimpleConfigurationSpace(values)
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
