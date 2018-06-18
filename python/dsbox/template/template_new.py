import datetime
import typing
import uuid
import yaml

import dateparser  # type: ignore
import jsonpath_ng  # type: ignore

from networkx import nx  # type: ignore
from d3m import exceptions, utils, index, runtime
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.pipeline import Pipeline, PipelineStep, StepBase, PrimitiveStep, PlaceholderStep, SubpipelineStep, ArgumentType, PlaceholderStep, Resolver, PIPELINE_SCHEMA_VALIDATOR
from d3m.primitive_interfaces.base import PrimitiveBaseMeta


from d3m.metadata.pipeline import Pipeline
from d3m.container.dataset import Dataset

argmentsmapper = {
    "container": ArgumentType.CONTAINER,
    "data": ArgumentType.DATA,
    "value": ArgumentType.VALUE,
    "primitive": ArgumentType.PRIMITIVE
}
# from .library import SemanticTypeDict

# default_template = {
#             "name": "my template",
#             "taskType": "",
#             "inputType": "",
#             # Steps have to be in execution order
#             "steps": [
#                 {
#                     "name": "my_step1",
#                     "primitives": [
#                         "dsbox.x.y",
#                         {
#                             "primitive": "dsbox.a.b",
#                             "hyperparameters": {
#                                 "x": 1
#                             }
#                         }
#                     ],
#                     "inputs": ["template_input"]
#                 },
#                 {
#                     "name": "my_step2",
#                     "primitives": ["sklearn.a.b"],
#                     "inputs": ["my_step1"]
#                 }

#             ]
#         }


class DSBoxTemplate():
    def __init__(self):
        self.primitive = index.search()

    def to_pipeline(self, configuration_point: typing.Dict[str, dict]) -> Pipeline:

        # configuration_point =
        # {
        #     "my_step1" : {
        #         "primitive": "dsbox.a.b",
        #         "hyperparameters": {
        #             "x": 1
        #         }
        #     },
        #     "my_step2" : {
        #         "primitive": "sklearn.a.b",
        #         "hyperparameters": {}
        #     }
        # }

        # do reasoning
        # binding = ....
        binding = {}
        for step in self.template["steps"]:
            # for k in configuration_point.keys():
            tmp = []
            tmp.append(configuration_point[step["name"]])
            binding[step["name"]] = tmp
        return self._to_pipeline(binding)

    def _to_pipeline(self, binding) -> Pipeline:
        # binding =
        # {
        #     "my_step1" : [
        #         {
        #             "primitive": "dsbox.c.d",
        #             "hyperparameters": {
        #                 "y": 3
        #             }
        #         },
        #         {
        #             "primitive": "dsbox.a.b",
        #             "hyperparameters": {
        #                 "x": 1
        #             }
        #         }
        #     ]
        #     ,
        #     "my_step2" : [
        #         {
        #             "primitive": "sklearn.a.b",
        #             "hyperparameters": {}
        #         }
        #     ]
        # }
        pipeline = Pipeline(name="Helloworld", context='PRETRAINING')  # generate empty pipeline with i/o/s/u =[]
        templateinput = pipeline.add_input("input dataset")
        outputs = {}
        for k in self.template["steps"]:
            name = k["name"]
            # primitiveStep = PrimitiveStep(binding[name]["primitive"].metatdata.query())
            for v in binding[name]:
                primitiveStep = PrimitiveStep(self.primitive[v["primitive"]].metadata.query())
                print("adding", v["primitive"])
                pipeline.add_step(primitiveStep)
                outputs[name] = primitiveStep.add_output("produce")
                if v["hyperparameters"] != {}:
                    hyper = v["hyperparameters"]
                    for n in hyper.keys():
                        print(n, ArgumentType.VALUE, hyper[n]["value"])
                        primitiveStep.add_hyperparameter(n, argmentsmapper[hyper[n]["type"]], hyper[n]["value"])
                        print(primitiveStep.hyperparams)
                    # pass
                if len(k["inputs"]) == 1:
                    for i in k["inputs"]:
                        if i == "template_input":
                            primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, templateinput)
                        else:
                            primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[i])
                elif len(k["inputs"]) == 2:
                    primitiveStep.add_argument("inputs", ArgumentType.CONTAINER, outputs[k["inputs"][0]])
                    primitiveStep.add_argument("outputs", ArgumentType.CONTAINER, outputs[k["inputs"][1]])
                else:
                    raise exceptions.InvalidArgumentValueError("Should be less than 3 arguments!")
                print(primitiveStep.outputs, primitiveStep.arguments)
        general_output = outputs[self.template["steps"][-1]["name"]]
        pipeline.add_output(general_output, "predictions of input dataset")
        return pipeline

    def check_compatibility(dataset: Dataset, problem_description: dict):
        """Returns True is template is compatible with dataset and problem"""
        # form problem_description see d3m.d3m.metadata.problem.parse_problem_description()
        # check tasktype and inputType
        pass

    def importance(dataset: Dataset, problem_description: dict):
        """Returns number from 0 to 10, Assume compaiblity has already been verified"""
        return 5
