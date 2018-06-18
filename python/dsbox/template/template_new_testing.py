import template_new
from d3m import utils, index
import json
primitive = index.search()
# print(primitive['d3m.primitives.datasets.Denormalize'].metadata.pretty_print())
mydsbox = template_new.DSBoxTemplate()
with open("user_defined_template_sample.json", "r") as g:
    template = json.load(g)
with open("user_defined_conf_sample.json", "r") as f:
    configuration_point = json.load(f)
# mypipeline = mydsbox.to_pipeline(configuration_point)
# print(configuration_point)
# print(type(configuration_point))
# print(typeof(configuration_point))
# print(mypipeline)
# template_new.printpipeline(mypipeline)
#
mytemplate = template_new.MyTemplate(template)
