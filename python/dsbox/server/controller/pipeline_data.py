"""The Python implementation of the GRPC pipeline.PipelineDataServicer server."""
import grpc

import pipeline_data_pb2
import pipeline_data_pb2_grpc

from dsbox.server.controller.session_handler import Session

class PipelineData(pipeline_data_pb2_grpc.PipelineDataServicer):
    def AddFeatures(self, request, context):
        """Add and remove features to/from datasets
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoveFeatures(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AddSamples(self, request, context):
        """Add and remove records (rows) to/from datasets
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RemoveSamples(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReplaceData(self, request, context):
        """Replace individual data points in a set
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Materialize(self, request, context):
        """Persist the dataset with modifications applied for future use
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TrainValidationSplit(self, request, context):
        """Deterministic split of a dataset into training a validation
        Filters out all but validation records or training records depending on is_validation
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Revert(self, request, context):
        """Revert the dataset to the original state
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def add_to_server(self, server):
        pipeline_data_pb2_grpc.add_PipelineDataServicer_to_server(self, server)
