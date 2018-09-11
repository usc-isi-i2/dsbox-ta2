from d3m.metadata.pipeline import Pipeline, PrimitiveStep

import typing
import pprint
import networkx as nx
import json
import os


def pipe2str(pipeline: Pipeline) -> typing.AnyStr:
    # TODO
    desc = list(map(lambda s: (s.primitive, s.hyperparams), pipeline.steps))
    return pprint.pformat(desc)
