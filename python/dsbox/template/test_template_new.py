
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
