import argparse
import os
import psutil
import signal
import sys
import time
import traceback

from dsbox.controller.controller import Controller
from dsbox.controller.config import DsboxConfig

start_time = time.time()
# wrote_result = False


class StdoutLogger(object):
    def __init__(self, f):
        self.terminal = sys.stdout
        self.log = f

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


class StderrLogger(object):
    def __init__(self, f):
        self.err = sys.stderr
        self.log = f

    def write(self, message):
        self.err.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def main(config: DsboxConfig, debug_mode=False):
    controller = Controller(development_mode=False)
    controller.initialize(config)
    config.set_start_time()

    def kill_child_processes():
        process_id = os.getpid()
        parent = psutil.Process(process_id)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()

    # Define signal handler to simulate evaluation timeout
    def force_exit(a_signal, frame):
        print('==== Times up ====')
        time_used = (time.time() - start_time) / 60.0
        print("The time used so far is {:0.2f} minutes.".format(time_used))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

            print('[INFO] Killing child processes', flush=True)
            kill_child_processes()
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # sys.exit(0) generates SystemExit exception, which may
            # be caught and ignored.

            # This os._exit() cannot be caught.
            # print('SIGNAL exiting {}'.format(configuration_file), flush=True)
            os._exit(0)

    timeout = config.timeout

    if not debug_mode:
        if timeout > 0:
            signal.signal(signal.SIGALRM, force_exit)
            signal.alarm(timeout)
        else:
            raise Exception('Negative timeout {}'.format(timeout))

    print("[INFO] Now in training process")
    controller.initialize_from_config_for_evaluation(config)
    status = controller.train()
    print("[INFO] Training Done")

    controller.write_training_results()

    time_used = (time.perf_counter() - config.start_time) / 60.0
    print("[INFO] The time used for running program is {:0.2f} minutes.".format(time_used))

    controller.shutdown()

    return status.value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Perform a TA2 search run. The configuration of the run is specified through shell environment variables.")
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode. Turn off timemout forced exit")
    args = parser.parse_args()

    config = DsboxConfig()
    config.load()

    print("Configuration:")
    print(config)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(os.path.join(config.log_dir, 'out.txt'), 'w')

    sys.stdout = StdoutLogger(f)
    sys.stderr = StderrLogger(f)

    result = main(config, args.debug)
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    f.close()

    os._exit(result)
