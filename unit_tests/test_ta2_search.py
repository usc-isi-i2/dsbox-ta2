import json
import os
import pathlib
import unittest


class TestTA2Search(unittest.TestCase):

    def setUp(self):
        # self.output_root_dir = pathlib.Path.home() / 'output'
        # self.output_root_dir = pathlib.Path('/nas/home/kyao/output/docker/seed/uu1_datasmash')
        self.output_root_dir = pathlib.Path('/nas/home/kyao/output/docker/seed/38_sick')
        self.pipelines_scored_dir = self.output_root_dir / 'pipelines_scored'
        self.pipelines_ranked_dir = self.output_root_dir / 'pipelines_ranked'
        self.predictions_dir = self.output_root_dir / 'predictions'
        self.scores_dir = self.output_root_dir / 'score'

        self.scored_pipelines = list(self.pipelines_scored_dir.glob('*.json'))
        self.ranked_pipelines = list(self.pipelines_ranked_dir.glob('*.json'))
        self.predictions = list(self.predictions_dir.glob('*.csv'))
        self.scores = list(self.scores_dir.glob('*.score.json'))

    @staticmethod
    def _valid_json(file_name):
        try:
            with open(file_name, 'r') as f:
                json.load(f)
        except Exception:
            return False
        return True

    def test_search_result_layout(self):
        dirs = ["pipelines_ranked", "pipelines_scored", "predictions", "score", "pipelines_searched"]
        for dir in dirs:
            self.assertEqual(os.path.exists(self.output_root_dir / dir), True, "{} not exist".format(dir))

    def test_pipeline_scored(self):
        self.assertGreater(len(self.scored_pipelines), 0, "No pipeline scored")

    def test_pipeline_ranked(self):
        self.assertGreater(len(self.ranked_pipelines), 0, "No pipeline ranked")

    # def test_pipelines_dir_split(self):
    #     if self.pipelines_ranked_dir.exists():
    #         self.assertLessEqual(len(self.ranked_pipelines), 20, "More than 20 pipelines in /pipelines_ranked")

    def test_predictions(self):
        self.assertGreater(len(self.predictions), 0, "No predictions generated")
        if self.predictions_dir.exists():
            self.assertEqual(len(self.ranked_pipelines), len(self.predictions), "Some pipelines failed to generate predictions")
        for file in self.predictions:
            self.assertGreater((self.predictions_dir / file).stat().st_size, 0, f"Prediction file is empty: {file}")

    def test_scores(self):
        self.assertGreater(len(self.scores), 0, "No scores generated")
        if self.scores_dir.exists():
            self.assertEqual(len(self.ranked_pipelines), len(self.scores), "Some pipelines failed to generate scores")
        for file in self.scores:
            self.assertTrue((self.scores_dir / file).stat().st_size > 0, f"Prediction file is empty: {file}")

    def test_pipelines_scored_valid(self):
        for pipeline_file_name in self.scored_pipelines:
            self.assertEqual(self._valid_json(self.pipelines_scored_dir / pipeline_file_name), True,
                             "{} pipeline json invalid".format(pipeline_file_name))

    def test_pipelines_ranked_valid(self):
        for pipeline_file_name in self.ranked_pipelines:
            self.assertEqual(self._valid_json(self.pipelines_ranked_dir / pipeline_file_name), True,
                             "{} pipeline json invalid".format(pipeline_file_name))

    def test_pipeline_id(self):
        for pipeline_file_name in self.scored_pipelines:
            with open((self.pipelines_scored_dir / pipeline_file_name), 'r') as f:
                obj = json.load(f)
            self.assertEqual(pipeline_file_name.stem, obj["id"],
                             "{} pipeline file name and id mismatch".format(pipeline_file_name))


if __name__ == '__main__':
    unittest.main()
