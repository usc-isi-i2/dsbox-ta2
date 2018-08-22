import unittest
import sys
import json
import os


class TestTA2Search(unittest.TestCase):

    def setUp(self):
        self.dirname = os.path.dirname(__file__)
        self.search_config = json.load(open(os.path.join(self.dirname, "resources/38_sick/search_config.json"), 'r'))
        self.output_root = os.path.dirname(self.search_config["pipeline_logs_root"])

    def test_search_result_layout(self):
        keys = ["pipeline_logs_root", "executables_root", "temp_storage_root"]
        for key in keys:
            self.assertEqual(os.path.exists(self.search_config[key]), True, "{} not exist".format(key))
