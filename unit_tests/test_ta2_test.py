import unittest
import json
import os
import pandas as pd


class TestTA2Test(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.test_config = json.load(open(os.path.join(self.dirname, "resources/38_sick/test_config.json"), 'r'))
        self.output_root = os.path.dirname(self.test_config["executables_root"])
        self.pipelines_considered_dir = os.path.join(self.output_root, "pipelines_considered")
        self.prediction_dir_root = self.test_config["results_root"]

        pipelines = [os.path.join(self.output_root, "pipelines", f) for f in
                     os.listdir(os.path.join(self.output_root, "pipelines")) if f.endswith(".json")]
        rank_lst = list()
        for pipeline in pipelines:
            pipeline_json = json.load(open(pipeline, 'r'))
            rank_lst.append((pipeline_json["pipeline_rank"], pipeline_json['id']))
        self.best_fitted_pipeline_id = min(rank_lst)[1]
        self.best_fitted_pipeline_rank = min(rank_lst)[0]

        self.prediction = pd.read_csv(
            os.path.join(self.output_root, "predictions", self.best_fitted_pipeline_id, "predictions.csv"))

        self.ground_truth = pd.read_csv(os.path.join(self.output_root, "SCORE", "targets.csv"))

    def test_ta2_test_result_layout(self):
        for key in ["results_root"]:
            self.assertEqual(os.path.exists(self.test_config[key]), True, "{} not exist".format(key))

    def test_fitted_pipeline_is_best(self):
        if os.path.exists(self.pipelines_considered_dir):
            for pipeline_name in os.listdir(self.pipelines_considered_dir):
                if pipeline_name.endswith(".json"):
                    pipeline_json = json.load(open(os.path.join(self.pipelines_considered_dir, pipeline_name), 'r'))
                    self.assertGreaterEqual(pipeline_json["pipeline_rank"], self.best_fitted_pipeline_rank,
                                            "{} pipeline in pipeline_considered has lower rank than best pipepline in pipelines".format(
                                                pipeline_name))

    def test_prediction_dir(self):
        self.assertEqual(os.path.exists(self.prediction_dir_root), True, "Prediction dir not exist")

    def test_prediction_file(self):
        prediction_dirs = [x for x in os.listdir(self.prediction_dir_root) if
                           os.path.isdir(os.path.join(self.prediction_dir_root, x))]
        self.assertEqual(len(prediction_dirs), 1, "more than 1 prediction generated for running single test")
        self.assertEqual(self.best_fitted_pipeline_id, prediction_dirs[0],
                         "pipeline used for predict is not same as best pipeline")

    def test_prediction_result_column_index(self):
        self.assertEqual(all(list(self.prediction.columns.values == self.ground_truth.columns.values)), True,
                         "prediction column index incorrect")

    def test_prediction_result_d3mIndex(self):
        self.assertEqual(all(list(self.prediction["d3mIndex"] == self.ground_truth["d3mIndex"])), True,
                         "prediction column index incorrect")
