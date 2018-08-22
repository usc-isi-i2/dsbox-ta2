import unittest
import json
import os


class TestConfigFormat(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.dataset_dir = os.path.join(self.dirname, 'resources/38_sick')
        self.search_config_keys = ["problem_schema", "problem_root", "dataset_schema", "training_data_root"]
        self.test_config_keys = ["problem_schema", "problem_root", "dataset_schema", "test_data_root"]

    def test_config_exist(self):
        json_files = [x for x in os.listdir(self.dataset_dir) if x.endswith(".json")]

        self.assertIn("search_config.json", json_files, "search_config.json missing")
        self.assertIn("test_config.json", json_files, "test_config.json missing")

    def test_config_keys(self):
        search_config = json.load(open(os.path.join(self.dataset_dir, "search_config.json"), 'r'))
        self.assertEqual(len(set(self.search_config_keys).intersection(search_config.keys())), 4, "search_config missing key")

        test_config = json.load(open(os.path.join(self.dataset_dir, "test_config.json"), 'r'))
        self.assertEqual(len(set(self.test_config_keys).intersection(test_config.keys())), 4, "test_config missing key")

    def test_config_path_exist(self):
        search_config = json.load(open(os.path.join(self.dataset_dir, "search_config.json"), 'r'))
        for k, v in search_config.items():
            if k in self.search_config_keys:
                self.assertEqual(os.path.exists(os.path.join(self.dataset_dir, v)), True,
                                 "path: {} not exist".format(os.path.join(self.dataset_dir, v)))

        test_config = json.load(open(os.path.join(self.dataset_dir, "test_config.json"), 'r'))
        for k, v in test_config.items():
            if k in self.test_config_keys:
                self.assertEqual(os.path.exists(os.path.join(self.dataset_dir, v)), True,
                                 "path: {} not exist".format(os.path.join(self.dataset_dir, v)))
