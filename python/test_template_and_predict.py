
# sample input
# python3 test_template_and_predict.py MuxinTA1ClassificationTemplate2 /muxin/data/38_sick/ --Score_path /muxin/data/38_sick/SCORE/


# import d3m utility
from d3m import utils, index
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata.pipeline import PrimitiveStep, ArgumentType
from d3m.metadata.problem import parse_problem_description, TaskType, TaskSubtype
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Metadata, ALL_ELEMENTS

# import evaluation packages
# remember to install it from https://gitlab.datadrivendiscovery.org/nist/nist_eval_output_validation_scoring
from d3m_outputs import Predictions

# import python packages
import argparse
import json
import os
import networkx
from importlib import reload

# import dsbox-ta2 patterns

# os.sys.path.insert(0, "/muxin/dsbox-ta2/python/")  # remember to change this to your own path
from dsbox.template.runtime import Runtime, add_target_columns_metadata
from dsbox.template.template import TemplatePipeline, TemplateStep, DSBoxTemplate
from dsbox.template.library import TemplateLibrary
from dsbox.template.configuration_space import DimensionName, ConfigurationSpace, SimpleConfigurationSpace, ConfigurationPoint
from dsbox.template.search import get_target_columns


def load_data(data_path, problem_path) -> tuple:
    '''
    load dataset metadata
    '''
    dataset = D3MDatasetLoader()
    if "file:" not in data_path:
        data_path = 'file://{dataset_path}'.format(dataset_path=os.path.abspath(data_path))
    with open(problem_path) as f:
        problem_doc = json.load(f)
        problem = Metadata(problem_doc)
    dataset = dataset.load(dataset_uri=data_path)
    dataset = add_target_columns_metadata(dataset, problem)
    return dataset, problem


def load_template(template_name) -> DSBoxTemplate:
    '''
    load one template by name
    '''
    return TemplateLibrary(run_single_template=template_name).templates[0]()


def generate_pipeline(template) -> TemplatePipeline:
    space = template.generate_configuration_space()
    point = space.get_point_using_first_value()

    return template.to_pipeline(point)


def run_pipeline_and_predict(pipeline, train_data, test_data, problem, outdir, dataset_path) -> str:
    '''
    Predicting and write results to a path
    '''
    pipeline_runtime = Runtime(pipeline, fitted_pipeline_id="", log_dir=outdir)
    pipeline_runtime.fit(inputs=[train_data])
    resID = problem.query(())["inputs"]["data"][0]["targets"][0]["resID"]
    test_length = test_data.metadata.query((resID, ALL_ELEMENTS))["dimension"]["length"]
    for v in range(0, test_length):
        types = test_data.metadata.query((resID, ALL_ELEMENTS, v))["semantic_types"]
        for t in types:
            if t == "https://metadata.datadrivendiscovery.org/types/TrueTarget":
                target_col_name = test_data.metadata.query((resID, ALL_ELEMENTS, v))["name"]
                break

    pipeline_runtime.produce(inputs=[test_data])
    prediction = pipeline_runtime.produce_outputs[-1]

    # prediction.to_csv(outdir)
    print(prediction.head())

    d3m_index = get_target_columns(test_data, problem)["d3mIndex"]
    d3m_index = d3m_index.reset_index().drop(columns=["index"])
    prediction_col_name = prediction.columns[-1]
    prediction["d3mIndex"] = d3m_index
    prediction = prediction[["d3mIndex", prediction_col_name]]
    prediction = prediction.rename(columns={prediction_col_name: target_col_name})
    if outdir == -1:
        outdir = "./results/" + dataset_path.split("/")[-2] + "/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        filename = "prediction.csv"
        totalpath = outdir + filename
    with open(totalpath, "w") as f:

        prediction.to_csv(f, index=False)
    print("prediction result wrote to", totalpath)
    return totalpath


def validate_prediction(score_path, prediction_path) -> Predictions:
    path_to_score_root = score_path
    p = Predictions(prediction_path, path_to_score_root)
    if p.is_valid():
        return p
    else:
        return None


def main(args):
    train_dataset = args.Dataset_path + "TRAIN/dataset_TRAIN/datasetDoc.json"
    train_problem = args.Dataset_path + "TRAIN/problem_TRAIN/problemDoc.json"
    test_dataset = args.Dataset_path + "TEST/dataset_TEST/datasetDoc.json"
    test_problem = args.Dataset_path + "TEST/problem_TEST/problemDoc.json"
    train_data, train_problem = load_data(train_dataset, train_problem)
    test_data, test_problem = load_data(test_dataset, test_problem)
    print("Training Dataset loaded with metadata: ")
    print(train_data)
    print("Test Dataset loaded with metadata: ")
    print(test_data)
    template = load_template(args.Template_name)
    # print(template.templates)
    print("template", template.template["name"], "loaded")
    # dir()
    pipeline = generate_pipeline(template)
    prediction_path = run_pipeline_and_predict(pipeline, train_data, test_data, test_problem, args.Output_dir, args.Dataset_path)
    if args.Score_path == -1:
        print("No scoring requirements, exit")
        return 1
    else:
        prediction = validate_prediction(args.Score_path, prediction_path)
        ground_truth_path = args.Score_path + "/targets.csv"
        if prediction:
            print("Prediction is valid")
            print("Results of prediction: ", prediction.score(ground_truth_path))
        else:
            print("Prediction is not validate!")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single Template")
    parser.add_argument("Template_name", help="Template name in library.py")
    parser.add_argument("Dataset_path", help="Dataset path: /data/your_dataset_directory")
    parser.add_argument("--Score_path", help="Score path: /data/your_dataset_directory/SCORE", default=-1)
    parser.add_argument("--Output_dir", help="Output directory", default=-1)

    args = parser.parse_args()
    res = main(args)
    os._exit(res)
