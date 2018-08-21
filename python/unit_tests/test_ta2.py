import unittest
import sys
import json
import os
dirname = os.path.dirname(__file__)

dataset_dir = os.path.join(dirname, 'resources/38_sick')


class TestTA2(unittest.TestCase):

    def setUp(self):
        search_config = json.load(open(os.path.join(dataset_dir, "search_config.json"), 'r'))
        for k, v in search_config.items():
            search_config[k] = os.path.join(dataset_dir, v)

        test_config = json.load(open(os.path.join(dataset_dir, "test_config.json"), 'r'))
        for k, v in test_config.items():
            test_config[k] = os.path.join(dataset_dir, v)

        write_results_time = 2
        if 'timeout' in search_config:
            timeout = int(search_config['timeout']) - write_results_time
        else:
            timeout = 10 - write_results_time
            search_config['timeout'] = timeout
        print(search_config, test_config)
