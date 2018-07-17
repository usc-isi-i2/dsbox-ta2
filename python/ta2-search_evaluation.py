#!/usr/bin/env python

"""
Command Line Interface for running the DSBox TA2 Search
"""
import time
start_time = time.clock()
from dsbox_dev_setup import path_setup
import argparse
import json
import os
import signal
import sys
import traceback
from pprint import pprint

from dsbox.controller.controller import Controller
from  dsbox.controller.controller import Status
import os
# controller = Controller(development_mode=True)

import getpass

def main(args):

    timeout = 0
    configuration_file = args.configuration_file
    debug = args.debug

    controller = Controller(development_mode=debug)

    with open(configuration_file) as data:
        config = json.load(data)

    if args.cpus > -1:
        config['cpus'] = args.cpus

    # Replace output directories
    if args.output_prefix is not None:
        for key in ['pipeline_logs_root', 'executables_root', 'temp_storage_root']:
            if not '/output/' in config[key]:
                print(
                    'Skipping. No "*/output/" for config[{}]={}.'.format(key, config[key]))
            else:
                suffix = config[key].split('/output/', 1)[1]
                config[key] = os.path.join(args.output_prefix, suffix)

    #os.system('clear')
    print('Using configuation:')
    pprint(config)

    if debug:
        print("[INFO] Now in development mode")
        controller.initialize_from_config(config)
        # controller.initialize_from_config_train_test(config)
    else:
        print("[INFO] Now in evaluation mode")
        controller.initialize_from_config_for_evaluation(config)
    
    if 'training_data_root' in config:
        print("[INFO] Now in training process")
        status = controller.train()
        print("[INFO] Training Done")
        #print("*+"*10)
    elif 'test_data_root' in config:
        print("[INFO] Now in testing process")
        status = controller.test()
        print("[INFO] Testing Done")
    else:
        status = Status.PROBLEM_NOT_IMPLEMENT
        print("[ERROR] Neither train or test root was given, the program will exit.")
    
    time_used = time.clock() - start_time
    print("[INFO] The time used for running program is",time_used,"seconds.")
    return status.value


if __name__ == "__main__":
    # get the pass in parameters here
    os.env['D3MTIMEOUT']


    args = parser.parse_args()

    print(args)

    result = main(args)
    os._exit(result)
