import unittest
import sys
import json
import os


class TestTA2Search(unittest.TestCase):

    def setUp(self):
        self.search_config = json.load(open("resources/38_sick/search_config.json", 'r'))
        print(self.search_config)
