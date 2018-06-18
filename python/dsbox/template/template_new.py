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


class MyTemplate(DSBoxTemplate):
    def __init__(self):
        DSBoxTemplate.__init__(self)
        self.template = {
            "name": "my template",
            "taskType": "",
            "inputType": "",
            "steps": [

                {
                    "name": "my_step1",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["template_input"]
                },
                {
                    "name": "my_step2",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step1"]
                },
                {
                    "name": "my_step3",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step2"]
                },

                {
                    "name": "my_step4",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step3"]
                },
                {
                    "name": "my_step5",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step4"]
                },

                {
                    "name": "my_step6",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step5"]
                },
                {
                    "name": "my_step7",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step6"]
                },
                {
                    "name": "my_step8",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step7"]
                },
                {
                    "name": "my_step9",
                    "primitives": ["d3m.primitives.datasets.Denormalize", "d3m.primitives.datasets.DatasetToDataFrame", "d3m.primitives.data.ColumnParser", "d3m.primitives.data.ExtractAttributes",
                                   "d3m.primitives.data.CastToType", "d3m.primitives.sklearn_wrap.SKImputer", "d3m.primitives.data.ExtractTargets", "d3m.primitives.common_primitives.RandomForestClassifier"],
                    "inputs": ["my_step6", "my_step8"]
                }
            ]
        }

    # @override
    def importance(datset, problem_description):
        return 7


def printpipeline(pipeline):
    print("id", pipeline.id)
    print("name", pipeline.name)
    print("context", pipeline.context)
    print("steps: ")
    for s in pipeline.steps:
        print(s.primitive)
    for s in pipeline.outputs:
        print(s)
    for s in pipeline.inputs:
        print(s)


t = MyTemplate()
p = t.to_pipeline({
    "my_step1": {
        "primitive": "d3m.primitives.datasets.Denormalize",
        "hyperparameters": {},
    },
    "my_step2": {
        "primitive": "d3m.primitives.datasets.DatasetToDataFrame",
        "hyperparameters": {}
    },
    "my_step3": {
        "primitive": "d3m.primitives.data.ColumnParser",
        "hyperparameters": {}
    },
    "my_step4": {
        "primitive": "d3m.primitives.data.ExtractAttributes",
        "hyperparameters": {}
    },
    "my_step5": {
        "primitive": "d3m.primitives.data.CastToType",
        "hyperparameters": {}
    },
    "my_step6": {
        "primitive": "d3m.primitives.sklearn_wrap.SKImputer",
        "hyperparameters": {}
    },
    "my_step7": {
        "primitive": "d3m.primitives.data.ExtractTargets",
        "hyperparameters": {}
    },
    "my_step8": {
        "primitive": "d3m.primitives.data.CastToType",
        "hyperparameters": {}
    },
    "my_step9": {
        "primitive": "d3m.primitives.common_primitives.RandomForestClassifier",
        "hyperparameters": {"n_estimators": {"type": "value", "value": 15}}}
}
)
printpipeline(p)
#
p.check()  # check if the pipeline is valid
# # print(p._context_to_json())
with open("temp.yaml", "w") as y:
    p.to_yaml_content(y)
