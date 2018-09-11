import datetime
import typing
import uuid
import yaml
import pprint
import dateparser  # type: ignore
import jsonpath_ng  # type: ignore

from networkx import nx  # type: ignore
from d3m import exceptions, utils, index
from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.pipeline import Pipeline, PipelineStep, StepBase, PrimitiveStep, PlaceholderStep, SubpipelineStep, ArgumentType, PlaceholderStep, Resolver, PIPELINE_SCHEMA_VALIDATOR
from d3m.primitive_interfaces.base import PrimitiveBaseMeta

from d3m.metadata.pipeline import Pipeline
from d3m.container.dataset import Dataset

import dsbox.template.runtime as runtime

myfile = open("temp.yaml", "r")
mytemplate = Pipeline.from_yaml_content(myfile)
runtime.Runtime(mytemplate)


# myp = index.search()
# myclassf = myp["d3m.primitives.common_primitives.RandomForestClassifier"].metadata.query()
# with open("one_primitive", "w") as myfile:
#     d = yaml.dump(myclassf)
#     myfile.write(d)
# print(myp["d3m.primitives.common_primitives.RandomForestClassifier"].metadata.pretty_print())
