#!/usr/bin/env python
# from dsbox_dev_setup import path_setup
# path_setup()

import argparse
import logging
import os
import pprint
import sys
import tempfile
import typing

import google
import grpc

import ta3ta2_api.utils as utils
from ta3ta2_api import core_pb2_grpc as cpg

from ta3ta2_api import problem_pb2
from ta3ta2_api import value_pb2

from ta3ta2_api.core_pb2 import HelloRequest
from ta3ta2_api.core_pb2 import SearchSolutionsRequest
from ta3ta2_api.core_pb2 import GetSearchSolutionsResultsRequest
from ta3ta2_api.core_pb2 import ScoreSolutionRequest
from ta3ta2_api.core_pb2 import GetScoreSolutionResultsRequest
from ta3ta2_api.core_pb2 import SolutionRunUser
from ta3ta2_api.core_pb2 import EndSearchSolutionsRequest
from ta3ta2_api.core_pb2 import DescribeSolutionRequest
from ta3ta2_api.core_pb2 import FitSolutionRequest
from ta3ta2_api.core_pb2 import GetFitSolutionResultsRequest
from ta3ta2_api.core_pb2 import ProduceSolutionRequest
from ta3ta2_api.core_pb2 import GetProduceSolutionResultsRequest
from ta3ta2_api.core_pb2 import ListPrimitivesRequest

from ta3ta2_api.value_pb2 import Value
# from ta3ta2_api.value_pb2 import ValueType

from ta3ta2_api.pipeline_pb2 import PipelineDescription

from ta3ta2_api.problem_pb2 import ProblemDescription
from ta3ta2_api.problem_pb2 import ProblemPerformanceMetric
# from ta3ta2_api.problem_pb2 import PerformanceMetric
from ta3ta2_api.problem_pb2 import Problem
from ta3ta2_api.problem_pb2 import TaskKeyword
from ta3ta2_api.problem_pb2 import ProblemInput
from ta3ta2_api.problem_pb2 import ProblemTarget

# from d3m.metadata.problem import parse_problem_description
import d3m.metadata.pipeline as d3m_pipeline
import d3m.metadata.problem as d3m_problem

from dsbox.controller.config import find_dataset_docs

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s -- %(message)s')
_logger = logging.getLogger(__name__)


# API_VERSION="2019.12.4"
API_VERSION="2020.1.9"
ALLOWED_VALUE_TYPES = [utils.ValueType.DATASET_URI, utils.ValueType.CSV_URI, utils.ValueType.RAW]

PRINT_REQUEST = True
DATASET_BASE_PATH = ''
DASET_DOCS = {}


def config_datasets(datasets, use_docker_server) -> typing.List:
    global DATASET_BASE_PATH, DATASET_DOCS
    # root_dir = '/data/dsbox/datasets_v4.0.0/'
    root_dir = '/data/dsbox/datasets_MIN_METADATA/datasets/'
    if use_docker_server:
        DATASET_BASE_PATH = '/input'
    elif datasets == 'seed_aug':
        DATASET_BASE_PATH = root_dir + 'seed_datasets_data_augmentation'
    elif datasets == 'seed':
        DATASET_BASE_PATH = root_dir + 'seed_datasets_current'
    else:
        DATASET_BASE_PATH = root_dir + 'seed_datasets_current'

    print('Using dataset base path:', DATASET_BASE_PATH)
    DATASET_DOCS = find_dataset_docs(DATASET_BASE_PATH, _logger)
    return DATASET_BASE_PATH


def get_problem_description(dataset_base_path, dataset_name: str, problem_type: str) -> dict:
    dataset_names = os.listdir(dataset_base_path)
    if dataset_name not in dataset_names:
        _logger.error('Cannot find %s. Possible names are %s', dataset_name, dataset_names)
        raise Exception(f'Cannot find {dataset_name}')
    if problem_type in ('TRAIN', 'TEST'):
        problem_filepath = os.path.join(dataset_base_path, dataset_name, problem_type, 'problem_'+problem_type,
                                        'problemDoc.json')
    else:
        problem_filepath = os.path.join(dataset_base_path, dataset_name, dataset_name + '_problem', 'problemDoc.json')
    if not os.path.exists(problem_filepath):
        raise Exception(f'Cannot find problem doc: {problem_filepath}')
    problem_filepath = 'file://' + problem_filepath
    return d3m_problem.Problem.load(problem_filepath)


class DatasetInfo():
    def __init__(self, id, dataset_base_path, task_keywords, metric, target_index, resource_id, column_index, column_name):
        self.id = id
        self.task_keywords = task_keywords
        self.metric = metric
        self.target_index = target_index
        self.resource_id = resource_id
        self.column_index = column_index
        self.column_name = column_name
        if dataset_base_path == '/input/':
            dataset_path = dataset_base_path
            self.dataset_uri = 'file://' + os.path.join(dataset_path, id + '_dataset', 'datasetDoc.json')
        else:
            dataset_path = os.path.join(dataset_base_path, id)
            self.dataset_uri = 'file://' + os.path.join(dataset_path, id + '_dataset', 'datasetDoc.json')
        # print(self.dataset_uri)

        # if not os.path.exists(self.dataset_path):
        #     raise Exception('Not able to find dataset path: ' + self.dataset_path)

    @classmethod
    def generate_info_dict(cls, dataset_base_path):
        datasets_info_dict = {}
        datasets_info_dict['38_sick'] = DatasetInfo(
            '38_sick', dataset_base_path,
            [problem_pb2.CLASSIFICATION, problem_pb2.BINARY, problem_pb2.TABULAR],
            problem_pb2.F1_MACRO,
            30, '0', 30, 'Class')
        datasets_info_dict['185_baseball'] = DatasetInfo(
            '185_baseball', dataset_base_path,
            [problem_pb2.CLASSIFICATION, problem_pb2.MULTICLASS, problem_pb2.TABULAR],
            problem_pb2.F1_MACRO,
            0, '0', 18, 'Hall_of_Fame')
        datasets_info_dict['LL0_1100_popularkids'] = DatasetInfo(
            'LL0_1100_popularkids', dataset_base_path,
            [problem_pb2.CLASSIFICATION, problem_pb2.MULTICLASS, problem_pb2.TABULAR],
            problem_pb2.F1_MACRO,
            0, '0', 7, 'Goals')
        datasets_info_dict['59_umls'] = DatasetInfo(
            '59_umls', dataset_base_path,
            [problem_pb2.LINK_PREDICTION, problem_pb2.GRAPH],
            problem_pb2.ACCURACY,
            0, '1', 4, 'linkExists')
        datasets_info_dict['22_handgeometry'] = DatasetInfo(
            '22_handgeometry', dataset_base_path,
            [problem_pb2.REGRESSION, problem_pb2.UNIVARIATE, problem_pb2.IMAGE],
            problem_pb2.MEAN_SQUARED_ERROR,
            0, '1', 2, 'WRISTBREADTH')
        datasets_info_dict['196_autoMpg'] = DatasetInfo(
            '196_autoMpg',  dataset_base_path,
            [problem_pb2.REGRESSION, problem_pb2.UNIVARIATE, problem_pb2.TABULAR],
            problem_pb2.MEAN_SQUARED_ERROR,
            0, '0', 8, 'class')
        return datasets_info_dict


class Client(object):
    '''
    This script is a dummy TA3 client the submits a bunch of messages to drive the TA2 pipeline creation process.

    Based on SRI's TA2 test client
    '''

    def __init__(self):
        # TA2 run search timeout limit
        self.time_bound_search = 10  # minutes

        # Search give pipeline timeout limit
        self.time_bound_run = 3  # minutes

        # Limit the number of search solutions return. If zero, then return all solutions
        # and stream back solutions are they are searched.
        self.rank_solutions_limit = 0

        self.scratch_dir = tempfile.TemporaryDirectory().name

    def main(self, argv):
        '''
        Main entry point for the TA2 test client
        '''

        _logger.info("Running TA2/TA3 Interface version %s", API_VERSION)

        # Standardized TA2-TA3 port is 45042
        address = 'localhost:45042'
        channel = grpc.insecure_channel(address)

        # Create the stub to be used in each message call
        stub = cpg.CoreStub(channel)

        parser = argparse.ArgumentParser(description='Dummy TA3 client')
        parser.add_argument('--docker', action='store_true',
                            help='The dataset path deepnds on if server is running on docker or not')
        parser.add_argument('--seed', action='store_const', const='seed', dest='datasets')
        parser.add_argument('--seed-aug', action='store_const', const='seed_aug', dest='datasets')

        parser.add_argument('--sick', action='store_const', const='38_sick', dest='dataset')
        parser.add_argument('--kids', action='store_const', const='LL0_1100_popularkids', dest='dataset')
        parser.add_argument('--baseball', action='store_const', const='185_baseball', dest='dataset')
        parser.add_argument('--dataset', dest='dataset')

        parser.add_argument('--list', action='store_true', help='List primitives')

        parser.add_argument('--basic', action='store_true')
        parser.add_argument('--complete', action='store_true')
        parser.add_argument('--solution', help='Describe solution')
        parser.add_argument('--produce', help='Call pipeline produce, given solution id')
        parser.add_argument('--fit', help='Call pipeline produce, given solution id')
        parser.add_argument('--end-search', help='End search, given search id')

        parser.add_argument('--time-bound-search', type=int, default=self.time_bound_search,
                            help='Time bound limit for TA2 search (default: %(default)s minutes)')
        parser.add_argument('--time-bound-run', type=int, default=self.time_bound_run,
                            help='Time bound limit for searching given pipeline (default: %(default)s minutes)')
        parser.add_argument('--rank_solutions_limit', type=int, default=self.rank_solutions_limit,
                            help='Limit the number of search solutions return. If zero, then return all solutions and'
                            'stream back solutions are they are searched.')

        parser.add_argument('--pipeline', help='Call SearchSolutions, give saved pipeline description json file')

        args = parser.parse_args()

        self.time_bound_search = args.time_bound_search
        self.time_bound_run = args.time_bound_run
        self.rank_solutions_limit = args.rank_solutions_limit

        datasets = 'seed'
        if args.datasets:
            datasets = args.datasets
        config_datasets(datasets, args.docker)
        train_problem_desc = get_problem_description(DATASET_BASE_PATH, args.dataset, 'TRAIN')
        test_problem_desc = get_problem_description(DATASET_BASE_PATH, args.dataset, 'TEST')

        pprint.pprint(train_problem_desc)

        # dataset_path = get_dataset_path(args.docker)
        # dataset_info_dict = DatasetInfo.generate_info_dict(dataset_path)
        # dataset_name = args.dataset if args.dataset else '38_sick'
        # dataset_info = dataset_info_dict[dataset_name]

        # print('dataset: ', dataset_info.dataset_uri)

        if args.basic:
            # Make a set of calls that follow the basic pipeline search
            search_id = self.basicPipelineSearch(stub, train_problem_desc, test_problem_desc)
            print('Search ID', search_id)
        elif args.complete:
            search_id = self.completePipelineSearch(stub, train_problem_desc, test_problem_desc)
            print('Search ID', search_id)
        elif args.solution:
            solution_id = args.solution
            self.describeSolution(stub, solution_id)
        elif args.produce:
            solution_id = args.produce
            # self.basicProduceSolution(stub, solution_id, train_problem_desc)
            self.basicProduceSolution(stub, solution_id, test_problem_desc)
        elif args.fit:
            solution_id = args.fit
            self.basicFitSolution(stub, solution_id, train_problem_desc)
        elif args.end_search:
            search_id = args.end_search
            self.endSearchSolutions(stub, search_id)
        elif args.list:
            self.listPrimitives(stub)
        elif args.pipeline:
            with open(args.pipeline, 'r') as fd:
                pipeline = d3m_pipeline.Pipeline.from_json(fd)
            self.execute_pipeline(stub, train_problem_desc, pipeline)
        else:
            print('Try adding --basic')

    def execute_pipeline(self, stub, problem: d3m_problem.Problem, pipeline: d3m_pipeline.Pipeline):

        searchSolutionsResponse = self.searchSolutionsPipeline(stub, problem, pipeline)

        search_id = searchSolutionsResponse.search_id

        solutions = self.processSearchSolutionsResultsResponses(stub, search_id)

    def searchSolutionsPipeline(self, stub, problem: d3m_problem.Problem, pipeline: d3m_pipeline.Pipeline):
        _logger.info("Call Search Solutions with pipeline: %s", pipeline.id)

        template = utils.encode_pipeline_description(pipeline, ALLOWED_VALUE_TYPES, self.scratch_dir)

        metric = problem['problem']['performance_metrics'][0]

        request = SearchSolutionsRequest(
            user_agent="Test Client",
            version=API_VERSION,
            time_bound_search=self.time_bound_search,
            priority=0,
            allowed_value_types=[value_pb2.RAW],
            problem=ProblemDescription(
                problem=Problem(
                    # id=problem['id'],
                    # version="3.1.2",
                    # name=problem['id'],
                    # description=problem['id'],
                    task_keywords=[x.value for x in problem['problem']['task_keywords']],
                    performance_metrics=[
                        ProblemPerformanceMetric(
                            # metric=problem['problem']['performance_metrics'][0]['metric'].value,
                            metric=metric['metric'].value,
                            k=metric.get('params', {}).get('k', None),
                            pos_label=metric.get('params', {}).get('pos_label', None),

                        )]
                    ),
                inputs=[ProblemInput(
                    dataset_id=problem['inputs'][0]['dataset_id'],
                    targets=[
                        ProblemTarget(
                            target_index=problem['inputs'][0]['targets'][0]['target_index'],
                            resource_id=problem['inputs'][0]['targets'][0]['resource_id'],
                            column_index=problem['inputs'][0]['targets'][0]['column_index'],
                            column_name=problem['inputs'][0]['targets'][0]['column_name']
                        )
                        ])]
                ),
            template=template, # TODO: We will handle pipelines later D3M-61
            inputs=[Value(dataset_uri='file://' + DATASET_DOCS[problem['inputs'][0]['dataset_id']]),],
            time_bound_run=self.time_bound_run, # in minutes
            rank_solutions_limit=self.rank_solutions_limit
        )
        print_request(request)
        reply = stub.SearchSolutions(request)
        log_msg(reply)
        return reply



    def basicPipelineSearch(self, stub, train_problem, test_problem, end_search=False):
        '''
        Follow the example on the TA2-TA3 API documentation that follows the basic pipeline
        search interation diagram.
        '''
        # 1. Say Hello
        self.hello(stub)

        # 2. Initiate Solution Search
        searchSolutionsResponse = self.searchSolutions(stub, train_problem)

        # 3. Get the search context id
        search_id = searchSolutionsResponse.search_id

        # 4. Ask for the current solutions
        solutions = self.processSearchSolutionsResultsResponses(stub, search_id)

        solution_id = None
        for count, solution in enumerate(solutions):
            _logger.info('solution #{}'.format(count))
            solution_id = solution.solution_id
            # break # for now lets just work with one solution

            # 5. Score the first of the solutions.
            scoreSolution = self.scoreSolutionRequest(stub, solution_id, test_problem)
            request_id = scoreSolution.request_id
            _logger.info("request id is: " + request_id)

            # 6. Get Score Solution Results
            scoreSolutionResults = self.getScoreSolutionResults(stub, request_id)

            # 7. Iterate over the score solution responses
            i = 0 # TODO: Strangely, having iterated over this structure in the getScoreSolutionResults method the
            # scoreSolutionResults shows as empty, hmmm
            for scoreSolutionResultsResponse in scoreSolutionResults:
                _logger.info("State of solution for run %s is %s" % (str(i), str(scoreSolutionResultsResponse.progress.state)))
                log_msg(scoreSolutionResultsResponse)
                i += 1

        if end_search:
            self.endSearchSolutions(stub, search_id)

        return search_id

    def completePipelineSearch(self, stub, train_problem, test_problem, end_search=False):
        '''
        Follow the example on the TA2-TA3 API documentation that follows the basic pipeline
        search interation diagram.
        '''
        # 1. Say Hello
        self.hello(stub)

        # 2. Initiate Solution Search
        searchSolutionsResponse = self.searchSolutions(stub, train_problem)

        # 3. Get the search context id
        search_id = searchSolutionsResponse.search_id

        # 4. Ask for the current solutions
        solutions = self.processSearchSolutionsResultsResponses(stub, search_id)

        solution_id = None
        for count, solution in enumerate(solutions):
            _logger.info('solution #{}'.format(count))
            solution_id = solution.solution_id

            self.describeSolution(stub, solution_id)

            _ = self.fitSolution(stub, solution_id, train_problem)

            scoreSolution = self.scoreSolutionRequest(stub, solution_id, train_problem)

            scoreSolutionResults = self.getScoreSolutionResults(stub, scoreSolution.request_id)

            i = 0
            for scoreSolutionResultsResponse in scoreSolutionResults:
                _logger.info("State of solution for run %s is %s" % (str(i), str(scoreSolutionResultsResponse.progress.state)))
                log_msg(scoreSolutionResultsResponse)
                i += 1

            self.basicProduceSolution(stub, solution_id, test_problem)

        if end_search:
            self.endSearchSolutions(stub, search_id)

        return search_id

    def basicFitSolution(self, stub, solution_id, problem):
        fit_solution_response = self.fitSolution(stub, solution_id, problem)

        get_fit_solution_results_response = self.getFitSolutionResults(stub, fit_solution_response.request_id)
        for fit_solution_results_response in get_fit_solution_results_response:
            log_msg(fit_solution_results_response)

    def basicProduceSolution(self, stub, solution_id, problem):
        produce_solution_response = self.produceSolution(stub, solution_id, problem)

        get_produce_solution_results_response = self.getProduceSolutionResults(stub, produce_solution_response.request_id)
        for produce_solution_results_response in get_produce_solution_results_response:
            log_msg(produce_solution_results_response)

    '''
    Invoke List Primitives
    '''
    def listPrimitives(self, stub):
        _logger.info('Listing primitives')
        request = ListPrimitivesRequest()
        reply = stub.ListPrimitives(request)
        for i, primitive in enumerate(reply.primitives):
            print(f'{i} {primitive}')

    '''
    Invoke Hello call
    '''
    def hello(self, stub):
        _logger.info("Calling Hello:")
        reply = stub.Hello(HelloRequest())
        log_msg(reply)


    '''
    Invoke Search Solutions
    Non streaming call
    '''
    def searchSolutions(self, stub, problem):
        _logger.info("Calling Search Solutions:")
        print([x.value for x in problem['problem']['task_keywords']])
        metric = problem['problem']['performance_metrics'][0]
        request = SearchSolutionsRequest(
            user_agent="Test Client",
            version=API_VERSION,
            time_bound_search=self.time_bound_search,
            priority=0,
            allowed_value_types=[value_pb2.RAW],
            problem=ProblemDescription(
                problem=Problem(
                    # id=problem['id'],
                    # version="3.1.2",
                    # name=problem['id'],
                    # description=problem['id'],
                    task_keywords=[x.value for x in problem['problem']['task_keywords']],
                    performance_metrics=[
                        ProblemPerformanceMetric(
                            # metric=problem['problem']['performance_metrics'][0]['metric'].value,
                            metric=metric['metric'].value,
                            k=metric.get('params', {}).get('k', None),
                            pos_label=metric.get('params', {}).get('pos_label', None),

                        )]
                    ),
                inputs=[ProblemInput(
                    dataset_id=problem['inputs'][0]['dataset_id'],
                    targets=[
                        ProblemTarget(
                            target_index=problem['inputs'][0]['targets'][0]['target_index'],
                            resource_id=problem['inputs'][0]['targets'][0]['resource_id'],
                            column_index=problem['inputs'][0]['targets'][0]['column_index'],
                            column_name=problem['inputs'][0]['targets'][0]['column_name']
                        )
                        ])]
                ),
            template=PipelineDescription(), # TODO: We will handle pipelines later D3M-61
            inputs=[Value(dataset_uri='file://' + DATASET_DOCS[problem['inputs'][0]['dataset_id']]),],
            time_bound_run=self.time_bound_run, # in minutes
            rank_solutions_limit=self.rank_solutions_limit
        )
        print_request(request)
        reply = stub.SearchSolutions(request)
        log_msg(reply)
        return reply


    '''
    Request and process the SearchSolutionsResponses
    Handles streaming reply from TA2
    '''
    def processSearchSolutionsResultsResponses(self, stub, search_id):
        _logger.info("Processing Search Solutions Result Responses:")
        request = GetSearchSolutionsResultsRequest(
            search_id=search_id
        )
        print_request(request)
        reply = stub.GetSearchSolutionsResults(request)

        results = []
        for searchSolutionsResultsResponse in reply:
            log_msg(searchSolutionsResultsResponse)
            results.append(searchSolutionsResultsResponse)
        return results


    '''
    For the provided Search Solution Results solution_id get the Score Solution Results Response
    Non streaming call
    '''
    def scoreSolutionRequest(self, stub, solution_id, problem):
        _logger.info("Calling Score Solution Request:")

        request = ScoreSolutionRequest(
            solution_id=solution_id,
            inputs=[Value(dataset_uri='file://' + DATASET_DOCS[problem['inputs'][0]['dataset_id']])],
            performance_metrics=[ProblemPerformanceMetric(
                metric=problem_pb2.ACCURACY
            )],
            users=[SolutionRunUser()],  # Optional so pushing for now
            configuration=None  # For future implementation
        )
        print_request(request)
        reply = stub.ScoreSolution(request)
        return reply


    '''
    For the provided Score Solution Results Response request_id score it against some data
    Handles streaming reply from TA2
    '''
    def getScoreSolutionResults(self, stub, request_id):
        _logger.info("Calling Score Solution Results with request_id: " + request_id)

        request = GetScoreSolutionResultsRequest(
            request_id=request_id
        )
        print_request(request)
        reply = stub.GetScoreSolutionResults(request)

        results = []

        # Iterating over yields from server
        for scoreSolutionResultsResponse in reply:
            log_msg(scoreSolutionResultsResponse)
            results.append(scoreSolutionResultsResponse)

        return results

    def endSearchSolutions(self, stub, search_id):
        _logger.info("Calling EndSearchSolutions with search_id: " + search_id)

        request = EndSearchSolutionsRequest(
            search_id=search_id
        )
        print_request(request)
        stub.EndSearchSolutions(request)

    def describeSolution(self, stub, solution_id):
        _logger.info("Calling DescribeSolution with solution_id: " + solution_id)
        request = DescribeSolutionRequest(solution_id=solution_id)
        print_request(request)
        reply = stub.DescribeSolution(DescribeSolutionRequest(
            solution_id=solution_id
        ))
        log_msg(reply)
        return reply

    def fitSolution(self, stub, solution_id, problem):
        _logger.info("Calling FitSolution with solution_id: " + solution_id)

        request = FitSolutionRequest(
            solution_id=solution_id,
            inputs=[Value(dataset_uri='file://' + DATASET_DOCS[problem['inputs'][0]['dataset_id']])],
            # expose_outputs = ['steps.7.produce'],
            expose_outputs=['outputs.0'],
            expose_value_types=[value_pb2.CSV_URI]
        )
        print_request(request)
        reply = stub.FitSolution(request)
        log_msg(reply)
        return reply

    def getFitSolutionResults(self, stub, request_id):
        _logger.info("Calling GetFitSolutionResults with request_id: " + request_id)
        request = GetFitSolutionResultsRequest(
            request_id=request_id
        )
        print_request(request)
        reply = stub.GetFitSolutionResults(request)
        log_msg(reply)
        return reply

    def produceSolution(self, stub, solution_id, problem):
        _logger.info("Calling ProduceSolution with solution_id: " + solution_id)

        request = ProduceSolutionRequest(
            fitted_solution_id=solution_id,
            inputs=[Value(dataset_uri='file://' + DATASET_DOCS[problem['inputs'][0]['dataset_id']])],
            # expose_outputs = ['steps.7.produce'],
            expose_outputs=['outputs.0'],
            expose_value_types=[value_pb2.CSV_URI]
        )
        print_request(request)
        reply = stub.ProduceSolution(request)
        log_msg(reply)
        return reply

    def getProduceSolutionResults(self, stub, request_id):
        _logger.info("Calling GetProduceSolutionResults with request_id: " + request_id)
        request = GetProduceSolutionResultsRequest(
            request_id=request_id
        )
        print_request(request)
        reply = stub.GetProduceSolutionResults(request)
        log_msg(reply)
        return reply


def to_dict(msg):
    '''
    convert request/reply to dict
    '''

    print('====', type(msg).__name__)
    fields = {}
    for field_descriptor, value in msg.ListFields():
        if callable(getattr(value, 'ListFields', None)):
            fields[field_descriptor.name] = to_dict(value)
        elif type(value) is google.protobuf.pyext._message.RepeatedCompositeContainer:
            fields[field_descriptor.name] = [to_dict(x) for x in value]
        else:
            fields[field_descriptor.name] = value
    result = { type(msg).__name__: fields}
    return result


def print_request(request):
    if PRINT_REQUEST:
        pprint.pprint(to_dict(request))


def log_msg(msg):
    '''
    Handy method for generating pipeline trace logs
    '''
    msg = str(msg)
    for line in msg.splitlines():
        _logger.info("    | %s" % line)
    _logger.info("    \\_____________")


'''
Entry point - required to make python happy
'''
if __name__ == "__main__":
    Client().main(sys.argv)
