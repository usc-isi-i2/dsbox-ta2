
import os
import d3m
import sys
os.sys.path.append("/nfs1/dsbox-repo/muxin/dsbox-ta2/python")

import library
from d3m.metadata.pipeline import Pipeline, PrimitiveStep, ArgumentType, PlaceholderStep
from importlib import reload
# import dsbox.template.template
# reload(dsbox.template.template)
import template
primitive = d3m.index.search()
# testing new features in template


def main():
    libdir = os.path.join(os.getcwd(), "../../library")
    std = library.SemanticTypeDict(libdir)
    print("reading")
    std.read_primitives()

    print(std.mapper)
    # print(primitive)

    mytemplate = template.TemplatePipeline(context='PRETRAINING')
    step_1 = PrimitiveStep(primitive['d3m.primitives.datasets.Denormalize'].metadata.query())  # what is the type of the input?
    mytemplate.add_step(step_1)
    step_2 = template.TemplateStep("classifier", "dsbox-classifiers")  # a k-v pair
    mytemplate.add_step(step_2)
    step_3 = template.TemplateStep("regressor", "dsbox-regressions")
    mytemplate.add_step(step_3)
    # print(mytemplate.template_nodes["modeller"].name)
    # print(mytemplate.template_nodes["modeller"].semantic_type)
    a = std.create_configuration_space(mytemplate)
    print(a)


main()
