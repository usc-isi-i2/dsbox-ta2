#!/usr/bin/env python

import time
start_time = time.clock()
import argparse
import json
import signal
import traceback
from pprint import pprint

from importlib import reload
import dsbox.controller.controller
reload(dsbox.controller.controller)
from dsbox.controller.controller import Controller
import os
controller = Controller(development_mode=True)

import getpass

def main(args):

    timeout = 0
    configuration_file = args.configuration_file
    debug = args.debug

    controller = Controller(development_mode=debug, run_single_template=True)

    with open(configuration_file) as data:
        config = json.load(data)

    if 'saving_folder_loc' not in config:
        output_location = "/nfs1/dsbox-repo/" + getpass.getuser() + "/dsbox-ta2/python/output/" + config['dataset_schema'].rsplit("/", 3)[-3]
        # output_location = "/nfs1/dsbox-repo/" + 'qasemi' + "/dsbox-ta2/python/output/" + config['dataset_schema'].rsplit("/", 3)[-3]
        config["temp_storage_root"] = output_location + "/temp"
        config["saving_folder_loc"] = output_location
        config["executables_root"] = output_location + "/executables"
        config["pipeline_logs_root"] = output_location + "/logs"
        config["saved_pipeline_ID"] = ""

    # Define signal handler to exit gracefully
    # Either on an interrupt or after a certain time
    def write_results_and_exit(a_signal, frame):
        # print('SIGNAL exit: {}'.format(configuration_file))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            # print('SIGNAL exit done reset signal {}'.format(configuration_file))

            controller.write_training_results()
            # print('SIGNAL exit done writing: {}'.format(
            #     configuration_file), flush=True)
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # sys.exit(0) generates SystemExit exception, which may
            # be caught and ignore.

            # This os._exit() cannot be caught.
            # print('SIGNAL exiting {}'.format(configuration_file), flush=True)
            os._exit(0)

    # Set timeout, alarm and signal handler
    if 'timeout' in config:
        # Timeout less 1 minute to give system chance to clean up
        timeout = int(config['timeout']) - 1
    if args.timeout > -1:
        # Overide config timeout
        timeout = args.timeout
    if timeout > 0:
        signal.signal(signal.SIGINT, write_results_and_exit)
        signal.signal(signal.SIGTERM, write_results_and_exit)
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(60*timeout)
    config['timeout'] = timeout

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


    if 'test_data_schema' in config and config['test_data_schema'] != '':
        print("[INFO] Test data config found! Will use given test data.")
        controller.initialize_from_config(config)
    else:
        print("[INFO] No test data config found! Will split the data.")
        controller.initialize_from_config_train_test(config)

    status = controller.train()
    print("*+"*10)
    status = controller.test()
    print("[INFO] Testing Done")
    time_used = time.clock() - start_time
    print("[INFO] The time used for running program is",time_used,"seconds.")
    return status.value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run DSBox TA2 system using json configuration file'
    )

    parser.add_argument('configuration_file',
                        help='D3M TA2 json configuration file')
    parser.add_argument('--timeout', action='store', type=int, default=-1,
                        help='Overide configuation timeout setting. In minutes.')
    parser.add_argument('--cpus', action='store', type=int, default=-1,
                        help='Overide configuation number of cpus usage setting')
    parser.add_argument('--output-prefix', action='store', default=None,
                        help='''Overide configuation output directories paths (pipeline_logs_root, executables_root, temp_storage_root).
                        Replace path prefix "*/output/" with argument''')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode. No timeout and no output redirection')

    args = parser.parse_args()

    print(args)

    result = main(args)
    os._exit(result)