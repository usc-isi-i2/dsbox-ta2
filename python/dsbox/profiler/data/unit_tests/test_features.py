import unittest
import sys
import json
sys.path.append('../')
from data_profile import profile_data

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        with open('./sources/gt.json') as data_file:    
            self.ground_truth = json.load(data_file)

        self.profiler_result = profile_data("./sources/testData.csv")


    def helper(self, prefix, field_name, gt_dict, pr_dict):
        """
        BFS search to compare all the items in the given ground truth and profiler result dictionaries.
        there is 1E-6 precision tolerance for floating value
        Parameters
        ----------
        prefix: the prefix name for the current field (from the root)
        field_name: the field to be checked in the given dictionaries.
        gt_dict: the ground truth dictionary
        pr_dict: the profiler result dictionary
        ----------
        """
        gt = gt_dict.get(field_name)
        pr = pr_dict.get(field_name)
        prefix += "/{}".format(field_name) 
        # end case: no sub-node anymore
        if (type(gt) != dict):
            if (type(gt) == float): # set 1E-6 tolerance for floating value
                self.assertEqual(round(gt,5), round(pr,5), 
                    "field {} value: {} does not match with ground truth value: {}".format(prefix, pr, gt))
            else:
                self.assertEqual(gt, pr, "field {} value: {} does not match with ground truth value: {}".format(prefix, pr, gt))
            return

        # general case:
        for field_name in gt:
            self.helper(prefix, field_name, gt, pr)


    def test_all(self):
        """
        test main function. Only check the if the existed items in ground_truth also exist in profiler result.
        For the items that only exists in profiler result, will be ignored and pass the test.
        """
        tested_fields = ["numeric_stats", "distinct", "frequent-entries", "length", "special_type", "missing"]
        for column_name in self.ground_truth:
            gt = self.ground_truth.get(column_name)
            pr = self.profiler_result.get(column_name)
            # to be tested field:
            for field_name in tested_fields:
                self.helper(column_name, field_name, gt, pr)
            


if __name__ == '__main__':
    unittest.main()