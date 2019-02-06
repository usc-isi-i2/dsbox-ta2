#!/usr/bin/env python
# from dsbox_dev_setup import path_setup
# path_setup()

import argparse
import grpc
import logging
import os
import pprint
import sys

import google

from ta3ta2_api import core_pb2_grpc as cpg

from ta3ta2_api import problem_pb2 as problem_pb2
from ta3ta2_api import value_pb2 as value_pb2

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

from ta3ta2_api.value_pb2 import Value
from ta3ta2_api.value_pb2 import ValueType

from ta3ta2_api.pipeline_pb2 import PipelineDescription

from ta3ta2_api.problem_pb2 import ProblemDescription
from ta3ta2_api.problem_pb2 import ProblemPerformanceMetric
from ta3ta2_api.problem_pb2 import PerformanceMetric
from ta3ta2_api.problem_pb2 import Problem
from ta3ta2_api.problem_pb2 import TaskType
from ta3ta2_api.problem_pb2 import TaskSubtype
from ta3ta2_api.problem_pb2 import ProblemInput
from ta3ta2_api.problem_pb2 import ProblemTarget


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s -- %(message)s')
_logger = logging.getLogger(__name__)


PRINT_REQUEST = False

def get_dataset_path(use_docker_server):
    if use_docker_server:
        dataset_base_path = '/input/'
    else:
        dataset_base_path = '/nfs1/dsbox-repo/data/datasets-v31/seed_datasets_current'
    print('Using dataset base path:', dataset_base_path)
    return dataset_base_path

class DatasetInfo():
    def __init__(self, id, dataset_base_path, task_type, task_subtype, metric, target_index, resource_id, column_index, column_name):
        self.id = id
        self.task_type = task_type
        self.task_subtype = task_subtype
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
            problem_pb2.CLASSIFICATION, problem_pb2.BINARY, problem_pb2.F1_MACRO,
            30, '0', 30, 'Class')
        datasets_info_dict['185_baseball'] = DatasetInfo(
            '185_baseball', dataset_base_path,
            problem_pb2.CLASSIFICATION, problem_pb2.MULTICLASS, problem_pb2.F1_MACRO,
            0, '0', 18, 'Hall_of_Fame')
        datasets_info_dict['LL0_1100_popularkids'] = DatasetInfo(
            'LL0_1100_popularkids', dataset_base_path,
            problem_pb2.CLASSIFICATION, problem_pb2.MULTICLASS, problem_pb2.F1_MACRO,
            0, '0', 7, 'Goals')
        datasets_info_dict['59_umls'] = DatasetInfo(
            '59_umls', dataset_base_path,
            problem_pb2.LINK_PREDICTION, problem_pb2.NONE, problem_pb2.ACCURACY,
            0, '1', 4, 'linkExists')
        datasets_info_dict['22_handgeometry'] = DatasetInfo(
            '22_handgeometry', dataset_base_path,
            problem_pb2.REGRESSION, problem_pb2.UNIVARIATE, problem_pb2.MEAN_SQUARED_ERROR,
            0, '1', 2, 'WRISTBREADTH')
        return datasets_info_dict


'''
This script is a dummy TA3 client the submits a bunch of messages to drive the TA2 pipeline creation process.

Based on SRI's TA2 test client
'''
class Client(object):

    '''
    Main entry point for the TA2 test client
    '''
    def main(self, argv):
        _logger.info("Running TA2/TA3 Interface version v2018.6.2");

        # Standardized TA2-TA3 port is 45042
        address = 'localhost:45042'
        channel = grpc.insecure_channel(address)

        # Create the stub to be used in each message call
        stub = cpg.CoreStub(channel)

        parser = argparse.ArgumentParser(description='Dummy TA3 client')
        parser.add_argument('--docker', action='store_true',
                            help='The dataset path deepnds on if server is running on docker or not')

        parser.add_argument('--sick', action='store_const', const='38_sick', dest='dataset')
        parser.add_argument('--kids', action='store_const', const='LL0_1100_popularkids', dest='dataset')
        parser.add_argument('--baseball', action='store_const', const='185_baseball', dest='dataset')
        parser.add_argument('--dataset', dest='dataset')

        parser.add_argument('--basic', action='store_true')
        parser.add_argument('--solution')
        parser.add_argument('--produce')
        parser.add_argument('--fit')
        parser.add_argument('--end-search')
        args = parser.parse_args()

        dataset_path = get_dataset_path(args.docker)
        dataset_info_dict = DatasetInfo.generate_info_dict(dataset_path)
        dataset_name = args.dataset if args.dataset else '38_sick'
        dataset_info = dataset_info_dict[dataset_name]

        print('dataset: ', dataset_info.dataset_uri)

        if args.basic:
            # Make a set of calls that follow the basic pipeline search
            search_id = self.basicPipelineSearch(stub, dataset_info)
            print('Search ID', search_id)
        elif args.solution:
            solution_id = args.solution
            self.describeSolution(stub, solution_id)
        elif args.produce:
            solution_id = args.produce
            self.basicProduceSolution(stub, solution_id, dataset_info)
        elif args.fit:
            solution_id = args.fit
            self.basicFitSolution(stub, solution_id, dataset_info)
        elif args.end_search:
            search_id = args.end_search
            self.endSearchSolutions(stub, search_id)
        else:
            print('Try adding --basic')


    '''
    Follow the example on the TA2-TA3 API documentation that follows the basic pipeline
    search interation diagram.
    '''
    def basicPipelineSearch(self, stub, dataset_info):
        # 1. Say Hello
        self.hello(stub)

        # 2. Initiate Solution Search
        searchSolutionsResponse = self.searchSolutions(stub, dataset_info)

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
            scoreSolution = self.scoreSolutionRequest(stub, solution_id, dataset_info)
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

        # # 8. Now that we have some results lets can the Search Solutions request
        # self.endSearchSolutions(stub, search_id)

        return search_id


    def basicFitSolution(self, stub, solution_id, dataset_info):
        fit_solution_response = self.fitSolution(stub, solution_id, dataset_info)

        get_fit_solution_results_response = self.getFitSolutionResults(stub, fit_solution_response.request_id)
        for fit_solution_results_response in get_fit_solution_results_response:
            log_msg(fit_solution_results_response)

    def basicProduceSolution(self, stub, solution_id, dataset_info):
        produce_solution_response = self.produceSolution(stub, solution_id, dataset_info)

        get_produce_solution_results_response = self.getProduceSolutionResults(stub, produce_solution_response.request_id)
        for produce_solution_results_response in get_produce_solution_results_response:
            log_msg(produce_solution_results_response)

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
    def searchSolutions(self, stub, dataset_info):
        _logger.info("Calling Search Solutions:")
        request = SearchSolutionsRequest(
            user_agent="Test Client",
            version="2018.7.7",
            time_bound=2, # minutes
            priority=0,
            allowed_value_types=[value_pb2.RAW],
            problem=ProblemDescription(
                problem=Problem(
                    id=dataset_info.id,
                    version="3.1.2",
                    name=dataset_info.id,
                    description=dataset_info.id,
                    task_type=dataset_info.task_type,
                    task_subtype=dataset_info.task_subtype,
                    performance_metrics=[
                        ProblemPerformanceMetric(
                            metric=dataset_info.metric,
                        )]
                    ),
                inputs=[ProblemInput(
                    dataset_id=dataset_info.id,
                    targets=[
                        ProblemTarget(
                            target_index=dataset_info.target_index,
                            resource_id=dataset_info.resource_id,
                            column_index=dataset_info.column_index,
                            column_name=dataset_info.column_name
                        )
                        ])]
                ),
            template=PipelineDescription(), # TODO: We will handle pipelines later D3M-61
            inputs=[Value(dataset_uri=dataset_info.dataset_uri)]
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
    def scoreSolutionRequest(self, stub, solution_id, dataset_info):
        _logger.info("Calling Score Solution Request:")

        request = ScoreSolutionRequest(
            solution_id=solution_id,
            inputs=[ Value(dataset_uri=dataset_info.dataset_uri)],
            performance_metrics=[ProblemPerformanceMetric(
                metric=problem_pb2.ACCURACY
            )],
            users=[SolutionRunUser()], # Optional so pushing for now
            configuration=None # For future implementation
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

    def fitSolution(self, stub, solution_id, dataset_info):
        _logger.info("Calling FitSolution with solution_id: " + solution_id)

        request = FitSolutionRequest(
            solution_id=solution_id,
            inputs=[Value(dataset_uri=dataset_info.dataset_uri)],
            # expose_outputs = ['steps.7.produce'],
            expose_outputs = ['outputs.0'],
            expose_value_types = [value_pb2.CSV_URI]
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

    def produceSolution(self, stub, solution_id, dataset_info):
        _logger.info("Calling ProduceSolution with solution_id: " + solution_id)

        request = ProduceSolutionRequest(
            fitted_solution_id=solution_id,
            inputs=[Value(dataset_uri=dataset_info.dataset_uri)],
            # expose_outputs = ['steps.7.produce'],
            expose_outputs = ['outputs.0'],
            expose_value_types = [value_pb2.CSV_URI]
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

'''
Handy method for generating pipeline trace logs
'''
def log_msg(msg):
    msg = str(msg)
    for line in msg.splitlines():
        _logger.info("    | %s" % line)
    _logger.info("    \\_____________")


'''
Entry point - required to make python happy
'''
if __name__ == "__main__":
    Client().main(sys.argv)
