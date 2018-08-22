import json
import os
import sys


class ConfigConstructor:

    @staticmethod
    def construct(dataset):
        dirname = os.path.abspath(os.path.dirname(__file__))

        dataset_dir = os.path.join(dirname, dataset)

        search_config = json.load(open(os.path.join(dirname, "search_config_default.json"), 'r'))
        test_config = json.load(open(os.path.join(dirname, "test_config_default.json"), 'r'))

        for k, v in search_config.items():
            search_config[k] = os.path.join(dataset_dir, v)

        with open(os.path.join(dataset_dir, "search_config.json"), 'w') as f:
            json.dump(search_config, f, indent=2)

        for k, v in test_config.items():
            test_config[k] = os.path.join(dataset_dir, v)

        with open(os.path.join(dataset_dir, "test_config.json"), 'w') as f:
            json.dump(test_config, f, indent=2)


if __name__ == '__main__':
    ConfigConstructor.construct(sys.argv[1])
