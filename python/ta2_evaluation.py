import time
start_time = time.clock()
import json
import os
import signal
import traceback
from pprint import pprint

from dsbox.controller.controller import Controller
from dsbox.controller.controller import Status


def main():
    timeout = 0
    if os.environ["D3MRUN"] == "search":
        config = json.load(open('/input/search_config.json', 'r'))
    else:
        config = json.load(open('/input/test_config.json', 'r'))

    config["cpus"] = os.environ["D3MCPU"]
    config["ram"] = os.environ["D3MRAM"]
    config["timeout"] = os.environ["D3MTIMEOUT"]

    controller = Controller(development_mode=False)

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
    if timeout > 0:
        signal.signal(signal.SIGINT, write_results_and_exit)
        signal.signal(signal.SIGTERM, write_results_and_exit)
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(60 * timeout)
    config['timeout'] = timeout

    print('Using configuation:')
    pprint(config)

    if 'training_data_root' in config:
        print("[INFO] Now in training process")
        controller.initialize_from_config_for_evaluation(config)
        status = controller.train()
        print("[INFO] Training Done")
        # print("*+"*10)
    elif 'test_data_root' in config:
        print("[INFO] Now in testing process")
        controller.initialize_from_test_config_for_evaluation(config)
        fitted_pipeline_id = json.load(open(os.environ["D3MTESTOPT"], 'r'))["fitted_pipeline_id"]
        status = controller.test_fitted_pipeline(fitted_pipeline_id=fitted_pipeline_id)
        print("[INFO] Testing Done")
    else:
        status = Status.PROBLEM_NOT_IMPLEMENT
        print("[ERROR] Neither train or test root was given, the program will exit.")

    time_used = time.clock() - start_time
    print("[INFO] The time used for running program is", time_used, "seconds.")
    return status.value


if __name__ == "__main__":

    result = main()
    os._exit(result)
