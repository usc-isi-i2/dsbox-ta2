import time
import json
import os
import signal
import traceback
from pprint import pprint

from dsbox.controller.controller import Controller
from dsbox.controller.controller import Status

start_time = time.time()

def main():
    timeout = 0
    if os.environ["D3MRUN"] == "search":
        config = json.load(open(os.path.join(os.environ["D3MINPUTDIR"], "search_config.json"), 'r'))
    else:
        config = json.load(open(os.path.join(os.environ["D3MINPUTDIR"], "test_config.json"), 'r'))

    config["cpus"] = os.environ["D3MCPU"]
    config["ram"] = os.environ["D3MRAM"]

    # Time to write results (in minutes)
    write_results_time = 3
    timeout = int(os.environ["D3MTIMEOUT"]) - write_results_time
    config["timeout"] = timeout

    controller = Controller(development_mode=False)

    # Define signal handler to exit gracefully
    def write_results_and_exit(a_signal, frame):
        print('==== Times up ====')
        time_used = (time.time() - start_time) / 60.0
        print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

            controller.write_training_results()
            print('==== Done cleaning up ====')
            time_used = (time.time() - start_time) / 60.0
            print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used))
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # sys.exit(0) generates SystemExit exception, which may
            # be caught and ignored.

            # This os._exit() cannot be caught.
            # print('SIGNAL exiting {}'.format(configuration_file), flush=True)
            os._exit(0)



    if timeout > 0:
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(60 * timeout)
    else:
        raise Exception('Negative timeout {}'.format(timeout))

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
        controller.initialize_from_config_for_evaluation(config)
        fitted_pipeline_id = json.load(open(os.environ["D3MTESTOPT"], 'r'))["fitted_pipeline_id"]
        status = controller.test_fitted_pipeline(fitted_pipeline_id=fitted_pipeline_id)
        print("[INFO] Testing Done")
    else:
        status = Status.PROBLEM_NOT_IMPLEMENT
        print("[ERROR] Neither train or test root was given, the program will exit.")

    time_used = (time.time() - start_time) / 60.0
    print("[INFO] The time used for running program is {:0.2f} minutes.".format(time_used))

    return status.value


if __name__ == "__main__":

    result = main()
    os._exit(result)
