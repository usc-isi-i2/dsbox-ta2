
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
# import networkx
import csv
# from importlib import reload


from dsbox.pipeline.fitted_pipeline import FittedPipeline
from dsbox.template.runtime import Runtime, add_target_columns_metadata
from dsbox.template.search import get_target_columns


def load_one_pipeline(path) -> tuple:
    '''
    read a pipeline, return its rank and pipeline
    '''

    with open(path, "r") as f:
        pipeline = json.load(f)
        return (pipeline["id"], pipeline["pipeline_rank"])


def load_for_dataset(path, res) -> list:
    '''
    load all the pipelines under pipeline_path and return a sorted list of them 
    '''

    res_loc = os.path.join(path, res)
    pipeline_loc = os.path.join(res_loc, "pipelines")

    pipelines_for_dataset = os.listdir(pipeline_loc)
    result = []
    if pipelines_for_dataset:
        print("All the pipelines for", res, "are", pipelines_for_dataset)
        print("*" * 20)
        for p in pipelines_for_dataset:
            tmp_path = os.path.join(pipeline_loc, p)
            result.append(load_one_pipeline(tmp_path))

    else:
        print("No fitted pipeline generated for", res_loc)
        return []
    result.sort(key=lambda tup: tup[1])
    return result


def load_all_fitted_pipeline(pipeline_path) -> dict:
    '''
    go through all the directorys from problem runs
    '''

    mypath = os.path.join(pipeline_path)
    allfile = os.listdir(mypath)
    pip_mapper = {}
    for res in allfile:
        if not res.endswith(".txt"):
            pip_mapper[res] = load_for_dataset(mypath, res)

    # print(allfile)
    return pip_mapper


def top_selection(mapper) -> dict:
    '''
    select top 10 pipelines from the dict of pipelines and also remove pipelines that are empty
    '''
    rmlist = []
    for key in mapper.keys():
        if len(mapper[key]) == 0:
            rmlist.append(key)
        if len(mapper[key]) >= TOP_NUM:
            mapper[key] = mapper[key][0:TOP_NUM]
    if rmlist:
        for r in rmlist:
            del mapper[r]
    return mapper


def create_fitted_pipelines_for_dataset(path, pipelines, log_dir) -> list:
    '''
    return a list of (fitted_pipeline, run)
    '''
    result = []
    for p in pipelines:
        result.append(FittedPipeline.load(folder_loc=path, pipeline_id=p[0], log_dir=log_dir))
    return result


def load_test_dataset_for_pipeline(config_path) -> tuple:
    '''
    load and return test_dataset and test_problem given by configfile: test_config.json
    '''
    test_config_path = os.path.join(config_path, "test_config.json")
    with open(test_config_path, "r") as f:
        test_config = json.load(f)
        data_path = test_config["dataset_schema"]
        problem_path = test_config["problem_schema"]
    dataset = D3MDatasetLoader()
    if "file:" not in data_path:
        data_path = 'file://{dataset_path}'.format(dataset_path=os.path.abspath(data_path))
    with open(problem_path) as f:
        problem_doc = json.load(f)
        problem = Metadata(problem_doc)
    dataset = dataset.load(dataset_uri=data_path)
    dataset = add_target_columns_metadata(dataset, problem)
    return dataset, problem


def predict_and_write(pipeline, test_dataset, test_problem, saving_path) -> str:
    '''
    run produce for pipelines and store as .csvs and return stored path
    '''
    resID = test_problem.query(())["inputs"]["data"][0]["targets"][0]["resID"]
    test_length = test_dataset.metadata.query((resID, ALL_ELEMENTS))["dimension"]["length"]
    for v in range(0, test_length):
        types = test_dataset.metadata.query((resID, ALL_ELEMENTS, v))["semantic_types"]
        for t in types:
            if t == "https://metadata.datadrivendiscovery.org/types/TrueTarget":
                target_col_name = test_dataset.metadata.query((resID, ALL_ELEMENTS, v))["name"]
                break

    pipeline.produce(inputs=[test_dataset])
    prediction = pipeline.runtime.produce_outputs[-1]
    d3m_index = get_target_columns(test_dataset, test_problem)["d3mIndex"]
    d3m_index = d3m_index.reset_index().drop(columns=["index"])
    prediction_col_name = prediction.columns[-1]
    prediction["d3mIndex"] = d3m_index
    prediction = prediction[["d3mIndex", prediction_col_name]]
    prediction = prediction.rename(columns={prediction_col_name: target_col_name})
    # print(prediction.head())
    with open(saving_path, "w") as f:
        prediction.to_csv(f, index=False)
    print("Prediction result wrote to", saving_path)
    return saving_path


def score_prediction(prediction_file, ground_truth_dir) -> dict:
    '''
    using NIST to score the result and return a dict that contain informations
    '''
    res = {}
    return res


def main(args):
    # print(args.path, args.filename)
    all_pipelines = load_all_fitted_pipeline(args.path)
    run_pipelines = top_selection(all_pipelines)
    TOP_NUM = args.top_n
    global TOP_NUM

    for dataset_pipeline in run_pipelines.keys():
        print("Start testing", dataset_pipeline)
        folder_path = os.path.join(args.path, dataset_pipeline)
        dataset, problem = load_test_dataset_for_pipeline(os.path.join(args.configs, dataset_pipeline))
        print("Using dataset", dataset, "and problem description", problem)
        fitted_pipelines = create_fitted_pipelines_for_dataset(folder_path, run_pipelines[dataset_pipeline], os.path.join(folder_path, "logs"))
        for fitted_pipeline, run in fitted_pipelines:
            '''
            sequence associate with rank
            '''
            # print(fitted_pipelines)
            fname = fitted_pipeline.id + ".csv"
            dir_path = os.path.join(folder_path, "results")
            # if not os.path.exists(dir_path):
            #     try:
            #         os.makedirs(os.path.dirname(dir_path))
            #     except:
            #         print("path not created.")
            os.makedirs(dir_path, exist_ok=True)
            saving_path = os.path.join(dir_path, fname)
            prediction_file = predict_and_write(fitted_pipeline, dataset, problem, saving_path)
            # score_prediction(prediction_file)
            # fitted_pipeline.produce(inputs =[dataset])
            # prediction = fitted_pipeline.produce_outputs[-1]
    # print(run_pipelines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all fitted pipeline and give results")
    parser.add_argument("--path", help="Where the pipelines stored, example: /nfs1/dsbox-repo/muxin/ta2-outputs/seed", default="/nfs1/dsbox-repo/muxin/ta2-outputs/seed")
    parser.add_argument("--configs", help="Where the configuration files stored, example: /nfs1/dsbox-repo/muxin/all_confs/seed", default="/nfs1/dsbox-repo/muxin/all_confs/seed")
    parser.add_argument("--filename", help="Name of the output csv", default=-1)
    parser.add_argument("--top_n", help="top n pipelines for testing", default=20)
    args = parser.parse_args()
    main(args)
