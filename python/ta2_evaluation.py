import time
import os
import psutil
import signal
import sys
import traceback

from dsbox.controller.controller import Controller
from dsbox.controller.config import DsboxConfig

start_time = time.time()


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


def main(config: DsboxConfig):
    controller = Controller(development_mode=False)

    def kill_child_processes():
        process_id = os.getpid()
        parent = psutil.Process(process_id)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()

    # Define signal handler to exit gracefully
    def write_results_and_exit(a_signal, frame):
        print('==== Times up ====')
        time_used = (time.time() - start_time) / 60.0
        print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used))
        try:
            # Reset to handlers to default as not to output multiple times
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

            print('[INFO] Killing child processes', flush=True)


            print('[INFO] writing results', flush=True)
            controller.write_training_results()

            print('==== Done cleaning up ====', flush=True)
            time_used = (time.time() - start_time) / 60.0
            print("[INFO] The time used so far is {:0.2f} minutes.".format(time_used), flush=True)

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

    write_results_time = 60  # seconds
    timeout = config.timeout - write_results_time

    if timeout > 0:
        signal.signal(signal.SIGALRM, write_results_and_exit)
        signal.alarm(timeout)
    else:
        raise Exception('Negative timeout {}'.format(timeout))

    print("[INFO] Now in training process")
    controller.initialize_from_config_for_evaluation(config)
    status = controller.train()
    print("[INFO] Training Done")

    time_used = (time.time() - start_time) / 60.0
    print("[INFO] The time used for running program is {:0.2f} minutes.".format(time_used))

    return status.value


if __name__ == "__main__":

    config = DsboxConfig()
    config.load()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(os.path.join(config.log_dir, 'out.txt'), 'w')

    sys.stdout = StdoutLogger(f)
    sys.stderr = StderrLogger(f)

    result = main(config)
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    f.close()

    os._exit(result)
