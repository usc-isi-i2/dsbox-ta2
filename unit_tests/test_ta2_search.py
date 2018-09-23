import unittest
import sys
import json
import os


class TestTA2Search(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.search_config = json.load(open(os.path.join(self.dirname, "resources/38_sick/search_config.json"), 'r'))
        self.output_root = os.path.dirname(self.search_config["pipeline_logs_root"])

        self.pipelines_dir = self.search_config["pipeline_logs_root"]
        self.pipelines_considered_dir = os.path.join(self.output_root, "pipelines_considered")
        self.executables_dir = self.search_config["executables_root"]
        self.supporting_files_dir = self.search_config["temp_storage_root"]

        self.executables_files = [x for x in os.listdir(self.executables_dir) if
                                  os.path.isfile(os.path.join(self.executables_dir, x))]
        self.pipeline_files = [x for x in os.listdir(self.pipelines_dir) if
                               os.path.isfile(os.path.join(self.pipelines_dir, x))]
        self.pickled_dir = [x for x in os.listdir(self.supporting_files_dir) if
                            os.path.isdir(os.path.join(self.supporting_files_dir, x)) and x != "logs"]

    @staticmethod
    def _valid_json(file_name):
        try:
            json.load(open(file_name, 'r'))
        except:
            return False
        return True

    def test_search_result_layout(self):
        keys = ["pipeline_logs_root", "executables_root", "temp_storage_root"]
        for key in keys:
            self.assertEqual(os.path.exists(self.search_config[key]), True, "{} not exist".format(key))

    def test_pipeline_generated(self):
        self.assertGreater(len(self.pipeline_files), 0, "No pipeline generated")

    def test_pipelines_dir_split(self):
        if os.path.exists(self.pipelines_considered_dir):
            self.assertLessEqual(len(self.pipeline_files), 20, "pipelines more than 20 in /pipelines")

    def test_pipelines_and_executables_dirs(self):
        self.assertEqual(self.executables_files, self.pipeline_files, "pipelines and executables file name mismatch")

    def test_pipelines_and_supporting_files_dir(self):
        trimed_json_suffix = sorted([x[:-5] for x in self.pipeline_files])
        self.assertEqual(sorted(self.pickled_dir), trimed_json_suffix, "pipelines and supporting files name mismatch")

    def test_pipelines_valid(self):
        for pipeline_file_name in self.pipeline_files:
            self.assertEqual(self._valid_json(os.path.join(self.pipelines_dir, pipeline_file_name)), True,
                             "{} pipeline json invalid".format(pipeline_file_name))

    def test_pipelines_considered_valid(self):
        if os.path.exists(self.pipelines_considered_dir):
            for pipeline_file_name in os.listdir(self.pipelines_considered_dir):
                if os.path.isfile(os.path.join(self.pipelines_considered_dir, pipeline_file_name)):
                    self.assertEqual(self._valid_json(os.path.join(self.pipelines_considered_dir, pipeline_file_name)),
                                     True,
                                     "{} considered pipeline json invalid".format(pipeline_file_name))

    def test_pipeline_id(self):
        for pipeline_file_name in self.pipeline_files:
            obj = json.load(open(os.path.join(self.pipelines_dir, pipeline_file_name), 'r'))
            self.assertEqual(pipeline_file_name[:-5], obj["id"],
                             "{} pipeline file name and id mismatch".format(pipeline_file_name))
