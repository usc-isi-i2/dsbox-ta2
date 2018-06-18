from d3m import exceptions, utils, index, runtime
import numpy as np
import typing
primitives = index.search()
i = (primitives["d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier"].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"])
o = (primitives["d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier"].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"])
# print(i and o)  # can use it  i and o = i
print(i == o)
print(i.__args__)
print(o)
print(o in i.__args__)
# print(len(i))
# print(len(o))
# print(type(i))
# print(type(o))
# i = (primitives["d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier"].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"])
# o = (primitives["d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier"].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"])
# print(type(i) == typing.Union)
# print(primitives["d3m.primitives.common_primitives.RandomForestClassifier"].metadata.query())
# check = np.zeros(shape=(len(primitives), len(primitives)))
# i = 0
# for v in primitives.keys():
#     j = 0
#     for u in primitives.keys():
#         if primitives[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"] == primitives[u].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"] or primitives[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"] in primitives[u].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"] or primitives[u].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"] in primitives[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"]:

#             check[i][j] = 1
#         j += 1
#     i += 1

#         else:
#             check[v][u] = 0
# print(v)primitives[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"]
# for v in primitives.keys():
#     print(v)
#     print("====inputs====", primitives[v].metadata.query()["primitive_code"]["class_type_arguments"]["Inputs"])
#     print("====outputs====", primitives["d3m.primitives.common_primitives.RandomForestClassifier"].metadata.query()["primitive_code"]["class_type_arguments"]["Outputs"])
# i, j...
# print(check)
# d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier
# ====inputs==== typing.Union[d3m.container.pandas.DataFrame, d3m.container.numpy.ndarray]
# ====outputs==== <class 'd3m.container.numpy.ndarray'>
# d3m.primitives.common_primitives.ImageReader
# ====inputs==== d3m.container.list.List[str]
# ====outputs==== <class 'd3m.container.numpy.ndarray'>
