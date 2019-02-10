import copy
import logging
import os
import pickle
import random
import string
import typing
import uuid

import pandas as pd
import numpy as np
from keras import backend as keras_backend
from pprint import pprint

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# ta3ta2_api = os.path.abspath(os.path.join(
#     CURRENT_DIR, '..', '..', '..', '..', 'ta3ta2-api'))
# print(ta3ta2_api)
# sys.path.append(ta3ta2_api)


import d3m
import d3m.metadata.base as mbase
import d3m.metadata.problem as d3m_problem
import d3m.container as d3m_container

from d3m.container.dataset import D3MDatasetLoader
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, SubpipelineStep
from d3m.primitive_interfaces.base import PrimitiveBase


from ta3ta2_api import core_pb2 as core_pb2
from ta3ta2_api import core_pb2_grpc as core_pb2_grpc
from ta3ta2_api import problem_pb2 as problem_pb2
from ta3ta2_api import value_pb2 as value_pb2
from ta3ta2_api import pipeline_pb2 as pipeline_pb2

from google.protobuf.reflection import GeneratedProtocolMessageType  # type: ignore
from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore

# import autoflowconfig
from ta3ta2_api.core_pb2 import DescribeSolutionResponse
from ta3ta2_api.core_pb2 import EndSearchSolutionsResponse
from ta3ta2_api.core_pb2 import EvaluationMethod
from ta3ta2_api.core_pb2 import FitSolutionResponse
from ta3ta2_api.core_pb2 import GetFitSolutionResultsResponse
from ta3ta2_api.core_pb2 import GetProduceSolutionResultsResponse
from ta3ta2_api.core_pb2 import GetScoreSolutionResultsResponse
from ta3ta2_api.core_pb2 import GetSearchSolutionsResultsResponse
from ta3ta2_api.core_pb2 import HelloResponse
from ta3ta2_api.core_pb2 import ListPrimitivesResponse
from ta3ta2_api.core_pb2 import PrimitiveStepDescription
from ta3ta2_api.core_pb2 import ProduceSolutionResponse
from ta3ta2_api.core_pb2 import Progress
from ta3ta2_api.core_pb2 import ProgressState
from ta3ta2_api.core_pb2 import Score
from ta3ta2_api.core_pb2 import ScoreSolutionResponse
from ta3ta2_api.core_pb2 import ScoringConfiguration
from ta3ta2_api.core_pb2 import SearchSolutionsResponse
from ta3ta2_api.core_pb2 import SolutionSearchScore
from ta3ta2_api.core_pb2 import StepDescription
from ta3ta2_api.core_pb2 import StepProgress
from ta3ta2_api.core_pb2 import SubpipelineStepDescription
from ta3ta2_api.core_pb2 import SolutionExportResponse

from ta3ta2_api.pipeline_pb2 import PipelineDescription
from ta3ta2_api.pipeline_pb2 import PipelineDescriptionInput
from ta3ta2_api.pipeline_pb2 import PipelineDescriptionOutput
from ta3ta2_api.pipeline_pb2 import PipelineDescriptionStep
from ta3ta2_api.pipeline_pb2 import PipelineDescriptionUser
from ta3ta2_api.pipeline_pb2 import PrimitivePipelineDescriptionStep
from ta3ta2_api.pipeline_pb2 import PrimitiveStepArgument
from ta3ta2_api.pipeline_pb2 import PrimitiveStepHyperparameter
from ta3ta2_api.pipeline_pb2 import StepOutput
from ta3ta2_api.pipeline_pb2 import ContainerArgument
from ta3ta2_api.pipeline_pb2 import DataArgument
from ta3ta2_api.pipeline_pb2 import PrimitiveArgument
from ta3ta2_api.pipeline_pb2 import ValueArgument
from ta3ta2_api.pipeline_pb2 import PrimitiveArguments

from ta3ta2_api.problem_pb2 import ProblemPerformanceMetric
from ta3ta2_api.problem_pb2 import PerformanceMetric
from ta3ta2_api.problem_pb2 import ProblemTarget

from ta3ta2_api.primitive_pb2 import Primitive

from ta3ta2_api.value_pb2 import Value
from ta3ta2_api.value_pb2 import ValueRaw
from ta3ta2_api.value_pb2 import ValueList
from ta3ta2_api.value_pb2 import ValueDict

from dsbox.controller.controller import Controller
from dsbox.controller.config import DsboxConfig
from dsbox.pipeline.fitted_pipeline import FittedPipeline

logging_level = logging.INFO

console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s -- %(message)s'))
console.setLevel(logging_level)
logging.getLogger('').addHandler(console)
logging.getLogger('').setLevel(logging_level)

_logger = logging.getLogger(__name__)

communication_value_types = [value_pb2.DATASET_URI, value_pb2.PICKLE_URI, value_pb2.PICKLE_BLOB, value_pb2.CSV_URI]


# self.controller = Controller()
# self.controller.output_directory = output_dir
# self.controller.initialize_from_ta3(config_dict)
# status = self.controller.train()
# self.search_solution_results[request.search_id] = self.controller.candidates
# _logger.info('    Found {} solutions.'.format(len(self.controller.candidates)))
# problem = self.controller.problem
# for solution in self.controller.candidates.values():
# fitted_pipeline = self.controller.load_fitted_pipeline(request.solution_id)

class TA2Servicer(core_pb2_grpc.CoreServicer):
    '''
    This class implements the CoreServicer base class. The CoreServicer defines the methods that must be supported by a
    TA2 server implementation. The CoreServicer class is generated by grpc using the core.proto descriptor file. See:
    https://gitlab.com/datadrivendiscovery/ta3ta2-api.
    '''

    def __init__(self, *, directory_mapping={}, fitted_pipeline_id: str = None, config: DsboxConfig = None) -> None:
        '''
        The __init__ method is used to establish the underlying TA2 libraries to service requests from the TA3 system.
        '''
        self.log_msg("Init invoked")
        self.controller = Controller(is_ta3=True)

        self.config = config
        # self.output_dir = config.output_dir
        # self.log_dir = os.path.join(self.config.output_dir, 'supporting_files', 'logs')
        self.directory_mapping = directory_mapping

        self.file_transfer_directory = os.path.join(self.config.output_dir, 'tmp')
        if not os.path.exists(self.file_transfer_directory):
            os.makedirs(self.file_transfer_directory)

        self.problem_parsed = {}
        # maps search solution id to config file
        self.search_solution = {}
        self.search_solution_results = {}

        self.score_solution = {}

        # maps fit solution id to fit solution request
        self.fit_solution: typing.Dict = {}

        # maps produce solution id to produce solution request
        self.produce_solution: typing.Dict = {}

        self._search_cache = {}

        if fitted_pipeline_id:
            fitted_pipeline = FittedPipeline.load(
                fitted_pipeline_id=fitted_pipeline_id, folder_loc=self.config.output_dir, log_dir=self.config.log_dir)
            self.fit_solution[fitted_pipeline_id] = fitted_pipeline

    def Hello(self, request, context):
        '''
        Hello call
        Non streaming call
        '''
        self.log_msg(msg="Hello invoked")
        # TODO: Figure out what we should be sending back to TA3 here.
        result = HelloResponse(user_agent="ISI",
                               version=core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version],
                               allowed_value_types=communication_value_types,
                               supported_extensions="")

        check(result)

        return result

    def SearchSolutions(self, request, context):
        '''
        Search Solutions call
        Non streaming
        '''
        self.log_msg(msg="SearchSolutions invoked")

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        if not request.version == core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version]:
            _logger.warning("Protocol Version does NOT match supported version {} != {}".format(
                core_pb2.DESCRIPTOR.GetOptions().Extensions[core_pb2.protocol_version], request.version))

        problem_json_dict = problem_to_json(request.problem)
        problem_parsed = problem_to_dict(request.problem)
        print('==json')
        pprint(problem_json_dict)
        print('==parsed')
        pprint(problem_parsed)

        self.problem_parsed = problem_parsed

        # Although called uri, it's just a filepath to datasetDoc.json
        self.dataset_uris = [input.dataset_uri for input in request.inputs]

        dataset_uris = [self._map_directories(uri) for uri in self.dataset_uris]

        # problem_config = DsboxConfig(self.config)
        # problem_config.update({
        #     'problem_json': problem_json_dict,
        #     'problem_parsed': problem_parsed,
        #     'dataset_schema': dataset_uri,
        #     'timeout': request.time_bound
        # })

        # convert to seconds
        self.config.timeout = request.time_bound * 60

        self.config.dataset_schema_files = dataset_uris
        self.config.set_problem(problem_json_dict, problem_parsed)

        print('===config')
        print(self.config)

        request_id = self.generateId()

        logger = self.config._logger
        self.config._logger = None
        self.search_solution[request_id] = copy.deepcopy(self.config)
        self.search_solution[request_id]._logger = logger
        self.config._logger = logger

        result = SearchSolutionsResponse(search_id=request_id)

        check(result)

        self.log_msg(msg="SearchSolutions returning")
        return result

    def GetSearchSolutionsResults(self, request, context):
        '''
        Get Search Solutions Results call
        Streams response to TA3
        '''
        self.log_msg(msg="GetSearchSolutionsResults invoked with search_id: " + request.search_id)

        if request.search_id not in self.search_solution:
            raise Exception('Search Solution ID not found ' +  request.search_id)

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        if request.search_id in self._search_cache:
            search_solutions_results = self._search_cache[request.search_id]
        else:
            problem_config = self.search_solution[request.search_id]

            self.controller.initialize_from_ta3(problem_config)
            status = self.controller.train()

            candidates = self.controller.get_candidates()
            self.search_solution_results[request.search_id] = candidates
            _logger.info('    Found {} solutions.'.format(len(candidates)))

            search_solutions_results = []
            problem = self.controller.get_problem()
            for solution in candidates.values():
                fitted_pipeline_id = solution['id']
                if 'test_metrics' in solution and solution['test_metrics'] is not None:
                    search_solutions_results.append(to_proto_search_solution_request(
                        problem, fitted_pipeline_id, solution['test_metrics']))

            check(search_solutions_results)

            self._search_cache[request.search_id] = search_solutions_results

        for solution in search_solutions_results:
            yield solution

        self.log_msg(msg="DONE: GetSearchSolutionsResults invoked with search_id: " + request.search_id)

    def ScoreSolution(self, request, context):
        '''
        Get the Score Solution request_id associated with the supplied solution_id
        Non streaming
        '''
        self.log_msg(msg="ScoreSolution invoked with solution_id: " + request.solution_id)

        request_id = self.generateId()
        result = ScoreSolutionResponse(
            request_id=request_id
        )

        self.score_solution[request_id] = request

        check(result)

        return result

    def GetScoreSolutionResults(self, request, context):
        '''
        Get Score Solution Results call
        Streams response to TA3
        '''
        self.log_msg(msg="GetScoreSolutionResults invoked with request_id: " + request.request_id)

        if request.request_id not in self.score_solution:
            raise Exception('Request id not found ' +  request.request_id)

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        score_request = self.score_solution[request.request_id]
        self.score_solution.pop(request.request_id, None)

        search_solutions_results = []

        fitted_pipeline_id = score_request.solution_id

        if score_request.inputs is not None:
            dataset_uri = score_request.inputs[0].dataset_uri
            if not self.dataset_uris[0] == dataset_uri:
                _logger.error("Dataset_uri not the same %s != %s", self.dataset_uris[0], dataset_uri)

        problem = self.controller.get_problem()
        for results in self.search_solution_results.values():
            for solution in results.values():
                if score_request.solution_id == solution['id']:
                    self.log_msg(msg='    Return search solution: {}'.format(solution))
                    fitted_pipeline_id = solution['id']
                    if 'test_metrics' in solution and solution['test_metrics'] is not None:
                        search_solutions_results.append(to_proto_score_solution_request(
                            problem, fitted_pipeline_id, solution['test_metrics']))

        check(search_solutions_results)

        self.log_msg('    Returning {} results'.format(len(search_solutions_results)))
        for solution in search_solutions_results:
            self.log_msg(msg=solution)
            yield solution

    def EndSearchSolutions(self, request, context):
        '''
        End the solution search process with the supplied search_id
        Non streaming
        '''
        self.log_msg(msg="EndSearchSolutions invoked with search_id: " + request.search_id)
        self.problem_parsed = {}
        self.search_solution.pop(request.search_id, None)
        self.search_solution_results.pop(request.search_id, None)

        return EndSearchSolutionsResponse()

    def SolutionExport(self, request, context):
        '''
        Exports a solution for evaluation based on NIST specifications.

        '''
        fitted_pipeline_id = request.fitted_solution_id
        self.log_msg(msg=f"SolutionExport invoked with rank {request.rank} fitted_solution_id {fitted_pipeline_id}")
        self.controller.export_solution(fitted_pipeline_id)

        return SolutionExportResponse()

    def StopSearchSolutions(self, request, context):
        '''
        Stops the search but leaves all currently found solutions available.
        '''
        # With current serial implementation, search is already stoped. Do nothing.
        return EndSearchSolutionsResponse()

    def ListPrimitives(self, request, context):
        primitives = []
        for python_path in d3m.index.search():
            primitives.append(to_proto_primitive(d3m.index.get_primitive(python_path)))
        return ListPrimitivesResponse(primitives=primitives)

    def ProduceSolution(self, request, context):
        self.log_msg(msg="ProduceSolution invoked with request_id " + request.fitted_solution_id)
        self.log_msg(request)

        request_id = self.generateId()
        self.produce_solution[request_id] = {
            'request' : request,
            'start' : Timestamp().GetCurrentTime()
        }
        return ProduceSolutionResponse(request_id=request_id)

    def GetProduceSolutionResults(self, request, context):
        self.log_msg(msg="GetProduceSolutionResults invoked with request_id " + request.request_id)

        if request.request_id not in self.produce_solution:
            raise Exception('Request id not found: ' + request.request_id)

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        produce_request = self.produce_solution[request.request_id]['request']
        start_time = self.produce_solution[request.request_id]['start']
        self.produce_solution.pop(request.request_id, None)

        fitted_pipeline_id = produce_request.fitted_solution_id

        # Load dataset
        loader = D3MDatasetLoader()
        dataset_uri = produce_request.inputs[0].dataset_uri
        dataset_uri = self._map_directories(dataset_uri)
        dataset = loader.load(dataset_uri=dataset_uri)

        # hack
        add_true_target(dataset)

        print('Load fitted pipeline', self.config.output_dir, fitted_pipeline_id)
        fitted_pipeline = FittedPipeline.load(fitted_pipeline_id=fitted_pipeline_id, folder_loc=self.config.output_dir, log_dir=self.config.log_dir)
        fitted_pipeline.produce(inputs=[dataset])

        timestamp = Timestamp()

        steps_progress = []
        for i, step in enumerate(fitted_pipeline.pipeline.steps):
            steps_progress.append(
                StepProgress(
                    progress=Progress(
                        state=core_pb2.COMPLETED,
                        status="Done",
                        start=start_time,
                        end=timestamp.GetCurrentTime())))

        step_outputs = {}
        for expose_output in produce_request.expose_outputs:
            parsed_output = parse_step_output(expose_output)
            if 'outputs' in parsed_output:
                parsed_output = parse_step_output(fitted_pipeline.pipeline.outputs[parsed_output['outputs']]['data'])
            dataframe = fitted_pipeline.get_produce_step_output(parsed_output['steps'])

            if 'outputs' in expose_output:
                if len(dataframe.columns) > 1:
                    print(dataframe.shape)
                    print(dataframe.columns)
                    print(dataframe.head())
                    filepath = to_csv_file(dataframe,
                                           self.file_transfer_directory,
                                           "produce_{}_{}".format(request.request_id, expose_output),
                                           index=False)
                else:
                    entry_id = find_entry_id(dataset)
                    if self.problem_parsed:
                        target_column_name = self.problem_parsed['inputs'][0]['targets'][0]['column_name']
                    else:
                        target_column_name = find_target_column_name(dataset, entry_id)
                    index_column_name, index_column = find_index_column_name_index(dataset, entry_id)
                    dataframe.columns = [target_column_name]
                    dataframe = pd.DataFrame(np.concatenate((dataset[entry_id].loc[:, [index_column_name]].as_matrix(), dataframe.as_matrix()), axis=1))
                    dataframe.columns = [index_column_name, target_column_name]
                    dataframe = dataframe.set_index(index_column_name)
                    filepath = to_csv_file(dataframe, self.file_transfer_directory, "produce_{}_{}".format(request.request_id, expose_output))
            step_outputs[expose_output] = Value(csv_uri=filepath)

        produce_solution_results = []
        produce_solution_results.append(GetProduceSolutionResultsResponse(
            progress=Progress(state=core_pb2.COMPLETED,
                              status="Done",
                              start=start_time,
                              end=timestamp.GetCurrentTime()),
            steps=steps_progress,
            exposed_outputs=step_outputs
        ))

        for result in produce_solution_results:
            yield result

    def FitSolution(self, request, context):
        self.log_msg(msg="FitSolution invoked with request_id " + request.solution_id)
        self.log_msg(request)

        request_id = self.generateId()
        self.fit_solution[request_id] = {
            'request': request,
            'start': Timestamp().GetCurrentTime()
        }
        return FitSolutionResponse(request_id=request_id)

    def GetFitSolutionResults(self, request, context):
        self.log_msg(msg="GetFitSolutionResults invoked with request_id " + request.request_id)

        if request.request_id not in self.fit_solution:
            raise Exception('Request id not found: ' + request.request_id)

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        fit_request = self.fit_solution[request.request_id]['request']
        start_time = self.fit_solution[request.request_id]['start']
        self.fit_solution.pop(request.request_id, None)

        fitted_pipeline_id = fit_request.solution_id

        # Load dataset
        loader = D3MDatasetLoader()
        dataset_uri = fit_request.inputs[0].dataset_uri
        dataset_uri = self._map_directories(dataset_uri)
        dataset = loader.load(dataset_uri=dataset_uri)

        # hack
        add_true_target(dataset)

        old_fitted_pipeline = FittedPipeline.load(fitted_pipeline_id=fitted_pipeline_id, folder_loc=self.config.output_dir, log_dir=self.config.log_dir)

        if old_fitted_pipeline.dataset_id == dataset.metadata.query(())['id']:
            # Nothigh to do. Old fitted pipeline was trained on the same dataset
            self.log_msg(msg="Reuse fitted pipeline")

            fit_solution_results = []
            fit_solution_results.append(GetFitSolutionResultsResponse(
                progress=Progress(state=core_pb2.COMPLETED,
                                  status="Done",
                                  start=start_time,
                                  end=Timestamp().GetCurrentTime()),
                steps=[],
                exposed_outputs=[],
                fitted_solution_id=old_fitted_pipeline.id
            ))

        else:
            self.log_msg(msg="Training new fitted pipeline")

            fitted_pipeline = FittedPipeline(old_fitted_pipeline.pipeline,
                                             dataset.metadata.query(())['id'],
                                             log_dir=self.config.log_dir,
                                             id=str(uuid.uuid4()),
                                             metric_descriptions=old_fitted_pipeline.metric_descriptions)

            fitted_pipeline.fit(inputs=[dataset])
            fitted_pipeline.produce(inputs=[dataset])

            fitted_pipeline.save(self.config.output_dir)

            timestamp = Timestamp()

            steps_progress = []
            for i, step in enumerate(fitted_pipeline.pipeline.steps):
                primitive_metadata = step.primitive.metadata.query()
                primitive_name = primitive_metadata['name']
                steps_progress.append(
                    StepProgress(
                        progress=Progress(
                            state=core_pb2.COMPLETED,
                            status="Done",
                            start=start_time,
                            end=timestamp.GetCurrentTime())))

            step_outputs = {}
            for expose_output in fit_request.expose_outputs:
                parsed_output = parse_step_output(expose_output)
                if 'outputs' in parsed_output:
                    parsed_output = parse_step_output(fitted_pipeline.pipeline.outputs[parsed_output['outputs']]['data'])
                dataframe = fitted_pipeline.get_fit_step_output(parsed_output['steps'])

                if 'outputs' in expose_output:
                    if len(dataframe.columns) > 1:
                        print(dataframe.shape)
                        print(dataframe.columns)
                        print(dataframe.head())
                        filepath = to_csv_file(dataframe,
                                               self.file_transfer_directory,
                                               "fit_{}_{}".format(request.request_id, expose_output),
                                               index=False)
                    else:
                        entry_id = find_entry_id(dataset)
                        if self.problem_parsed:
                            target_column_name = self.problem_parsed['inputs'][0]['targets'][0]['column_name']
                        else:
                            target_column_name = find_target_column_name(dataset, entry_id)
                        index_column_name, index_column = find_index_column_name_index(dataset, entry_id)
                        dataframe.columns = [target_column_name]
                        dataframe = pd.DataFrame(np.concatenate((dataset[entry_id].loc[:, [index_column_name]].as_matrix(), dataframe.as_matrix()), axis=1))
                        dataframe.columns = [index_column_name, target_column_name]
                        dataframe = dataframe.set_index(index_column_name)

                        filepath = to_csv_file(dataframe, self.file_transfer_directory, "fit_{}_{}".format(request.request_id, expose_output))
                step_outputs[expose_output] = Value(csv_uri=filepath)

            fit_solution_results = []
            fit_solution_results.append(GetFitSolutionResultsResponse(
                progress=Progress(state=core_pb2.COMPLETED,
                                  status="Done",
                                  start=start_time,
                                  end=timestamp.GetCurrentTime()),
                steps=steps_progress,
                exposed_outputs=step_outputs,
                fitted_solution_id=fitted_pipeline.id
            ))

        # Return results
        for result in fit_solution_results:
            yield result

    def UpdateProblem(self, request, context):
        _logger.error("UpdateProblem not yet implemented")
        pass

    def DescribeSolution(self, request, context) -> DescribeSolutionResponse:
        self.log_msg(msg="DescribeSolution invoked with soution_id " + request.solution_id)

        # Workaround for loading in keras graphs multiple times
        keras_backend.clear_session()

        fitted_pipeline = self.controller.load_fitted_pipeline(request.solution_id)

        pipeline = fitted_pipeline.pipeline

        result = DescribeSolutionResponse(
            pipeline=to_proto_pipeline(pipeline, fitted_pipeline.id),
            steps=to_proto_steps_description(pipeline)
        )

        check(result)

        return result

    def SaveSolution(self, request, context):
        _logger.error("SaveSolution not yet implemented")

    def LoadSolution(self, request, context):
        _logger.error("LoadSolution not yet implemented")

    def SaveFittedSolution(self, request, context):
        _logger.error("SaveFittedSolution not yet implemented")

    def LoadFittedSolution(self, request, context):
        _logger.error("LoadFittedSolution not yet implemented")

    def log_msg(self, msg):
        '''
        Handy method for generating pipeline trace logs
        '''
        msg = str(msg)
        for line in msg.splitlines():
            _logger.info("    | %s" % line)
        _logger.info("    \\_____________")

    def generateId(self):
        '''
        Convenience method for generating 22 character id's
        '''
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(22))

    def _map_directories(self, uri):
        '''
        Map file path. For debugging outside the container environment.
        '''
        for host_dir, container_dir in self.directory_mapping.items():
            if 'file://'+container_dir in uri:
                _logger.debug('replace uri: %s', uri)
                uri = uri.replace('file://'+container_dir, 'file://'+host_dir)
                _logger.debug('  with: %s', uri)
                return uri
        return uri


def add_true_target(dataset):
    from d3m.metadata.base import ALL_ELEMENTS as AE
    rid = find_entry_id(dataset)
    col_num = dataset.metadata.query((rid, AE))['dimension']['length'] - 1
    column_metadata = dict(dataset.metadata.query((rid, AE, col_num)))
    types = column_metadata['semantic_types']
    types = tuple(x if x != 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
                  else 'https://metadata.datadrivendiscovery.org/types/TrueTarget'
                  for x in types)
    column_metadata['semantic_types'] = types
    dataset.metadata = dataset.metadata.update((rid, AE, col_num), column_metadata)


def find_entry_id(dataset):
    entry_id = '0'
    for resource_id in dataset.keys():
        if "https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint" in dataset.metadata.query((resource_id,))['semantic_types']:
            entry_id = resource_id
            break
    return entry_id


def find_target_column_name(dataset, entry_id):
    target_idx = dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS))['dimension']['length'] - 1
    for col_idx in range(dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS))['dimension']['length']):
        semantic_types = dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS, col_idx))['semantic_types']
        if ("https://metadata.datadrivendiscovery.org/types/Target" in semantic_types
            or "https://metadata.datadrivendiscovery.org/types/TrueTarget" in semantic_types):
            target_idx = col_idx
            break
    return dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS, target_idx))['name']


def find_index_column_name_index(dataset, entry_id):
    target_idx = 0
    for col_idx in range(dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS))['dimension']['length']):
        semantic_types = dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS, col_idx))['semantic_types']
        if "https://metadata.datadrivendiscovery.org/types/PrimaryKey" in semantic_types:
            target_idx = col_idx
            break
    return dataset.metadata.query((entry_id, mbase.ALL_ELEMENTS, target_idx))['name'], target_idx


def to_csv_file(dataframe, file_transfer_directory, file_prefix: str, *, index=True) -> str:
    file_path = os.path.join(file_transfer_directory, file_prefix + '.csv')
    dataframe.to_csv(file_path, index=index)
    return file_path


def to_pickle_blob(container) -> bytes:
    if isinstance(container, d3m_container.DataFrame):
        return pickle.dumps(d3m_container.pandas.dataframe_serializer(container))
    elif isinstance(container, d3m_container.ndarry):
        return pickle.dumps(d3m_container.pandas.ndarray_serializer(container))
    elif isinstance(container, d3m_container.List):
        return pickle.dumps(d3m_container.pandas.list_serializer(container))
    elif isinstance(container, d3m_container.Dataset):
        return pickle.dumps(d3m_container.pandas.dataset_serializer(container))
    else:
        raise Exception('Container type not recognized: {}'.format(type(container)))


def to_pickle_file(container, file_transfer_directory, file_prefix: str) -> str:
    file_path = os.path.join(file_transfer_directory, file_prefix + '.pkl')
    with open(file_path, 'rb') as out:
        if isinstance(container, d3m_container.DataFrame):
            pickle.dump(d3m_container.pandas.dataframe_serializer(container), out)
        elif isinstance(container, d3m_container.ndarry):
            pickle.dump(d3m_container.pandas.ndarray_serializer(container), out)
        elif isinstance(container, d3m_container.List):
            pickle.dump(d3m_container.pandas.list_serializer(container), out)
        elif isinstance(container, d3m_container.Dataset):
            pickle.dump(d3m_container.pandas.dataset_serializer(container), out)
        else:
            raise Exception('Container type not recognized: {}'.format(type(container)))
    return file_path


def parse_step_output(output_reference: str) -> dict:
    # E.g., output_reference=='steps.3.produce' or output_reference=='outputs.0'
    parts = output_reference.split('.')
    if len(parts) == 2 and parts[0] == 'outputs':
        return {'outputs' : int(parts[1])}
    elif len(parts) ==3 and parts[0] in ['step', 'steps']:
        return {'steps': int(parts[1]), 'method': parts[2]}
    else:
        raise Exception('Pipeline output reference not supported: ' + output_reference)


# The output of this function should be the same as the output for
# d3m/metadata/problem.py:parse_problem_description
def problem_to_dict(problem) -> typing.Dict:
    performance_metrics = []
    for metrics in problem.problem.performance_metrics:
        if metrics.metric == 0:
            d3m_metric = None
        else:
            d3m_metric = d3m_problem.PerformanceMetric(metrics.metric)
        params = {}
        if d3m_metric == d3m_problem.PerformanceMetric.F1:
            if metrics.pos_label is None or metrics.pos_label == '':
                params['pos_label'] = '1'
            else:
                params['pos_label'] = metrics.pos_label
        if metrics.k > 0:
            params['k'] = metrics.k
        performance_metrics.append({
            'metric' : d3m_metric,
            'params' : params
        })

    description: typing.Dict[str, typing.Any] = {
        'schema': d3m_problem.PROBLEM_SCHEMA_VERSION,
        'problem': {
            'id': problem.problem.id,
            # "problemVersion" is required by the schema, but we want to be compatible with problem
            # descriptions which do not adhere to the schema.
            'version': problem.problem.version,
            'name': problem.problem.name,
            'task_type': d3m_problem.TaskType(problem.problem.task_type),
            'task_subtype': d3m_problem.TaskSubtype(problem.problem.task_subtype),
            'performance_metrics': performance_metrics
        },
        # Not Needed
        # 'outputs': {
        #     'predictions_file': problem_doc['expectedOutputs']['predictionsFile'],
        # }
    }

    inputs = []
    for input in problem.inputs:
        dataset_id = input.dataset_id
        for target in input.targets:
            targets = []
            targets.append({
                'target_index': target.target_index,
                'resource_id': target.resource_id,
                'column_index': target.column_index,
                'column_name': target.column_name,
                'clusters_number': target.clusters_number
            })
        inputs.append({
            'dataset_id': dataset_id,
            'targets': targets
        })
    description['inputs'] = inputs

    return description


# The output of this function should be the same as the problemDoc.json
def problem_to_json(problem) -> typing.Dict:
    performance_metrics = []
    for metrics in problem.problem.performance_metrics:
        if metrics.metric == 0:
            d3m_metric = None
        else:
            d3m_metric = d3m_problem.PerformanceMetric(metrics.metric)
        params = {}
        if d3m_metric == d3m_problem.PerformanceMetric.F1:
            if metrics.pos_label is None or metrics.pos_label == '':
                params['pos_label'] = '1'
            else:
                params['pos_label'] = metrics.pos_label
        if metrics.k > 0:
            params['k'] = metrics.k
        ametric = {}
        if d3m_metric:
            ametric['metric'] =  d3m_metric.unparse()
        if params:
            ametric['params'] = params
        performance_metrics.append(ametric)

    description: typing.Dict[str, typing.Any] = {
        'about': {
            'problemSchemaVersion': problem.problem.version,
            'problemID': problem.problem.id,
            # "problemVersion" is required by the schema, but we want to be compatible with problem
            # descriptions which do not adhere to the schema.
            # 'version': problem.problem.version,
            'problemName': problem.problem.name,
            'taskType': d3m_problem.TaskType(problem.problem.task_type).unparse(),
            'taskSubtype': d3m_problem.TaskSubtype(problem.problem.task_subtype).unparse(),
        }
        # Not Needed
        # 'outputs': {
        #     'predictions_file': problem_doc['expectedOutputs']['predictionsFile'],
        # }
    }

    data = []
    for input in problem.inputs:
        dataset_id = input.dataset_id
        for target in input.targets:
            targets = []
            targets.append({
                'targetIndex': target.target_index,
                'resID': str(target.resource_id),
                'colIndex': target.column_index,
                'colName': target.column_name,
                'clustersNumber': target.clusters_number
            })
        data.append({
            'datasetID': dataset_id,
            'targets': targets
        })

    description['inputs'] = {
        'data': data,
        'performanceMetrics': performance_metrics
    }

    return description


def check(message, *, depth=0):
    pass

    # if depth==0:
    #     print('====Begin Check')
    # if message is None:
    #     raise Exception('None value')
    # elif type(message) in [list, tuple] or 'RepeatedComposite' in str(type(message)):
    #     for i, value in enumerate(message):
    #         print(' ' * (4*depth), 'index=', i)
    #         check(value, depth=depth+1)
    # elif isinstance(message,dict) or 'MessageMap' in str(type(message)):
    #     for key, value in message.items():
    #         print(' ' * (4*depth), key)
    #         check(value, depth=depth+1)
    # elif '_pb2' in str(type(message)):
    #     print(' ' * (4*depth), message.DESCRIPTOR.full_name)
    #     for (field_descriptor, field_value) in message.ListFields():
    #         if field_value is None:
    #             print(' ' * (4*depth), 'ERROR: check found None value:', field_descriptor.full_name)
    #         else:
    #             print(' ' * (4*depth), field_descriptor.full_name)
    #             check(field_value, depth=depth+1)
    # else:
    #     print(' ' * (4*depth), message)
    # if depth==0:
    #     print('====End Check')


def to_proto_value_raw(value):
    if value is None:
        return ValueRaw(null=value_pb2.NULL_VALUE)
    elif isinstance(value, int):
        return ValueRaw(int64=value)
    elif isinstance(value, float):
        return ValueRaw(double=value)
    elif isinstance(value, bool):
        return ValueRaw(bool=value)
    elif isinstance(value, str):
        return ValueRaw(string=value)
    elif isinstance(value, bytes):
        return ValueRaw(bytes=value)
    elif isinstance(value, list) or isinstance(value, tuple):
        alist = []
        for x in value:
            alist.append(to_proto_value_raw(x))
        return ValueRaw(list=ValueList(items=alist))
    elif isinstance(value, dict):
        adict = {}
        for k, v in value.items():
            adict[k] = to_proto_value_raw(v)
        return ValueRaw(dict=ValueDict(items=adict))
    else:
        raise ValueError('to_proto_value: Unknown value type {}({})'.format(type(value), value))


def to_proto_primitive(primitive_base: PrimitiveBase) -> Primitive:
    """
    Convert d3m Primitive to protocol buffer Prmitive
    """
    metadata = primitive_base.metadata.query()
    return Primitive(
        id=metadata['id'],
        version=metadata['version'],
        python_path=metadata['python_path'],
        name=metadata['name'],
        digest=metadata['digest'] if 'digest' in metadata else None
    )


def to_proto_primitive_step(step: PrimitiveStep) -> PipelineDescriptionStep:
    """
    Convert d3m PrimitiveStep to protocol buffer PipelineDescriptionStep
    """
    arguments = {}
    for argument_name, argument_desc in step.arguments.items():
        if argument_desc['type'] == mbase.ArgumentType.CONTAINER:
            # mbase.ArgumentType.CONTAINER
            arguments[argument_name] = PrimitiveStepArgument(
                container=ContainerArgument(data=argument_desc['data']))
        else:
            # mbase.ArgumentType.DATA
            arguments[argument_name] = PrimitiveStepArgument(
                data=DataArgument(data=argument_desc['data']))
    outputs = [StepOutput(id=output) for output in step.outputs]
    hyperparams = {}
    for name, hyperparam_dict in step.hyperparams.items():
        hyperparam_type = hyperparam_dict['type']
        hyperparam_data = hyperparam_dict['data']
        if hyperparam_type == mbase.ArgumentType.CONTAINER:
            hyperparam = PrimitiveStepHyperparameter(container=ContainerArgument(data=hyperparam_data))
        elif hyperparam_type == mbase.ArgumentType.DATA:
            hyperparam = PrimitiveStepHyperparameter(data=DataArgument(data=hyperparam_data))
        elif hyperparam_type == mbase.ArgumentType.PRIMITIVE:
            hyperparam = PrimitiveStepHyperparameter(primitive=PrimitiveArgument(data=hyperparam_data))
        elif hyperparam_type == mbase.ArgumentType.VALUE:
            hyperparam = PrimitiveStepHyperparameter(value=ValueArgument(data=Value(raw=to_proto_value_raw(hyperparam_data))))
        else:
            # Dataset is not a valid ArgumentType
            # Should never get here.
            raise ValueError('to_proto_primitive_step: invalid hyperparam type {}'.format(hyperparam_type))
        hyperparams[name] = hyperparam
    primitive_description = PrimitivePipelineDescriptionStep(
        primitive=to_proto_primitive(step.primitive),
        arguments=arguments,
        outputs=outputs,
        hyperparams=hyperparams,
        users=[PipelineDescriptionUser(id=user_description)
               for user_description in step.users]
    )
    return PipelineDescriptionStep(primitive=primitive_description)


def to_proto_pipeline(pipeline: Pipeline, id: str = None) -> PipelineDescription:
    """
    Convert d3m Pipeline to protocol buffer PipelineDescription
    """
    inputs = []
    outputs = []
    steps = []
    users = []
    for input_description in pipeline.inputs:
        if 'name' in input_description:
            inputs.append(PipelineDescriptionInput(name=input_description['name']))
    for output_description in pipeline.outputs:
        outputs.append(
            PipelineDescriptionOutput(
                name=output_description['name'] if 'name' in output_description else None,
                data=output_description['data']))
    for step in pipeline.steps:
        if isinstance(step, PrimitiveStep):
            step_description = to_proto_primitive_step(step)
        elif isinstance(step, SubpipelineStep):
            # TODO: Subpipeline not yet implemented
            # PipelineDescriptionStep(pipeline=pipeline_description)
            pass
        else:
            # TODO: PlaceholderStep not yet implemented
            #PipelineDescriptionStep(placeholder=placeholde_description)
            pass
        steps.append(step_description)
    for user in pipeline.users:
        users.append(PipelineDescriptionUser(
            id=user['id'],
            reason=user['reason'] if 'reason' in user else None,
            rationale=user['rationale'] if 'rationale' in user else None
        ))
    if id is None:
        id = pipeline.id

    pipeline_context = pipeline.context.value
    return PipelineDescription(
        id=id,
        source=pipeline.source,
        created=Timestamp().FromDatetime(pipeline.created.replace(tzinfo=None)),
        context=pipeline_context,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        name=pipeline.name,
        description=pipeline.description,
        users=users
    )


def to_proto_search_solution_request(problem, fitted_pipeline_id, metrics_result) -> typing.List[GetSearchSolutionsResultsResponse]:

    # search_solutions_results = []

    timestamp = Timestamp()

    # Todo: controller needs to remember the partition method
    scoring_config = ScoringConfiguration(
        method=core_pb2.HOLDOUT,
        train_test_ratio=5,
        random_seed=4676,
        stratified=True)
    targets = []
    problem_dict = problem
    for inputs_dict in problem_dict['inputs']:
        for target in inputs_dict['targets']:
            targets.append(ProblemTarget(
                target_index = target['target_index'],
                resource_id = target['resource_id'],
                column_index = target['column_index'],
                column_name = target['column_name'],
                clusters_number = target['clusters_number']))
    score_list = []
    for metric in metrics_result:
        ppm = ProblemPerformanceMetric(metric=d3m_problem.PerformanceMetric.parse(metric['metric']).name)
        if 'k' in metric:
            ppm = metric['k']
        if 'pos_label' in metric:
            ppm = metric['pos_label']
        score_list.append(Score(
            metric=ppm,
            fold=0,
            targets=targets,
            value=Value(raw=to_proto_value_raw(metric['value']))))
    scores = []
    scores.append(
        SolutionSearchScore(
            scoring_configuration=scoring_config,
            scores=score_list))
    result = GetSearchSolutionsResultsResponse(
        progress=Progress(state=core_pb2.COMPLETED,
                          status="Done",
                          start=timestamp.GetCurrentTime(),
                          end=timestamp.GetCurrentTime()),
        done_ticks=0, # TODO: Figure out how we want to support this
        all_ticks=0, # TODO: Figure out how we want to support this
        solution_id=fitted_pipeline_id, # TODO: Populate this with the pipeline id
        internal_score=0,
        # scores=None # Optional so we will not tackle it until needed
        scores=scores
    )

    return result


def to_proto_score_solution_request(problem, fitted_pipeline_id, metrics_result) -> typing.List[GetSearchSolutionsResultsResponse]:

    # search_solutions_results = []

    timestamp = Timestamp()

    # Todo: controller needs to remember the partition method
    scoring_config = ScoringConfiguration(
        method=core_pb2.HOLDOUT,
        train_test_ratio=5,
        random_seed=4676,
        stratified=True)
    targets = []
    problem_dict = problem
    for inputs_dict in problem_dict['inputs']:
        for target in inputs_dict['targets']:
            targets.append(ProblemTarget(
                target_index=target['target_index'],
                resource_id=target['resource_id'],
                column_index=target['column_index'],
                column_name=target['column_name'],
                clusters_number=target['clusters_number']))
    score_list = []
    for metric in metrics_result:
        ppm = ProblemPerformanceMetric(metric=d3m_problem.PerformanceMetric.parse(metric['metric']).name)
        if 'k' in metric:
            ppm = metric['k']
        if 'pos_label' in metric:
            ppm = metric['pos_label']
        score_list.append(Score(
            metric=ppm,
            fold=0,
            targets=targets,
            value=Value(raw=to_proto_value_raw(metric['value']))))
    result = GetScoreSolutionResultsResponse(
        progress=Progress(state=core_pb2.COMPLETED,
                          status="Done",
                          start=timestamp.GetCurrentTime(),
                          end=timestamp.GetCurrentTime()),
        scores=score_list
    )

    return result


def to_proto_steps_description(pipeline: Pipeline) -> typing.List[StepDescription]:
    '''
    Convert free hyperparameters in d3m pipeline steps to protocol buffer StepDescription
    '''
    # Todo: To be implemented
    descriptions: typing.List[StepDescription] = []
    return descriptions

    # for step in pipeline.steps:
    #     print(step)
    #     if isinstance(step, PrimitiveStep):
    #         free = step.get_free_hyperparms()
    #         values = {}
    #         for name, hyperparam_class in free.items():
    #             default = hyperparam_class.get_default()
    #             values[name] = to_proto_value_with_type(default, hyperparam_class.structural_type)
    #         descriptions.append(StepDescription(
    #             primitive=PrimitiveStepDescription(hyperparams=values)))
    #     else:
    #         # TODO: Subpipeline not yet implemented
    #         pass
    # return descriptions
