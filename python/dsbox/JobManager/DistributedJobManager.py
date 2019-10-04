import copy
import os
import logging
import psutil
import time
import threading
import typing

from enum import Enum
from math import ceil
from multiprocessing import Pool, Queue, Manager, current_process
from threading import Timer

_logger = logging.getLogger(__name__)
# used to save all PID of workers created
m = Manager()
_current_work_pids = m.list()

class TimerResponse(Enum):
    KILL_WORKERS = 0
    STOP_WORKER_JOBS = 1


class WorkerQueueHandler(logging.handlers.QueueHandler):
    '''
    Adds process name to log records
    '''
    def __init__(self, queue):
        super().__init__(queue)

    def prepare(self, record):
        if record is not None and record.msg is not None:
            record.msg = f'{current_process().name:17} > ' + record.msg
        return super().prepare(record)

    # def emit(self, record):
    #     print('emit:', record)
    #     return super().emit(record)

    # def enqueue(self, record):
    #     print('enqueue:', record)
    #     return super().enqueue(record)


class DistributedJobManager:
    def __init__(self, proc_num: int = 4, timer_response=TimerResponse.STOP_WORKER_JOBS):

        self.start_time = time.perf_counter()
        self.proc_num = proc_num
        self.timer_response = timer_response
        self._timeout_sec = -1

        self.manager = Manager()
        # self.manager.start()
        self.arguments_queue: Queue = self.manager.Queue()
        self.result_queue: Queue = self.manager.Queue()
        self.log_queue: Queue = self.manager.Queue()

        self.argument_lock = self.manager.Lock()
        self.result_lock = self.manager.Lock()

        # initialize
        self.job_pool: Pool = None

        self.timer: Timer = None

        # status counter
        self.ongoing_jobs: int = 0

        # start the workers
        self._start_workers(DistributedJobManager._posted_job_wrapper)

    @property
    def timeout_sec(self):
        return self._timeout_sec

    @timeout_sec.setter
    def timeout_sec(self, value: int):
        self._timeout_sec = value
        self._setup_timeout_timer()

    def _start_workers(self, target_method: typing.Callable):
        # Start logging listener
        lp = threading.Thread(target=DistributedJobManager._logger_thread, args=(self.log_queue,))
        lp.start()

        self.job_pool = Pool(processes=self.proc_num)
        self.job_pool.map_async(
            func=DistributedJobManager._internal_worker_process,
            iterable=[
                (self.arguments_queue, self.result_queue, target_method,
                 self.log_queue, DistributedJobManager._log_configurer)
                for a in range(self.proc_num)]
        )
        self.job_pool.close()  # prevents any additional worker to be added to the pool

    @staticmethod
    def _posted_job_wrapper(target_obj: typing.Any, target_method: str,
                            kwargs: typing.Dict = {}) -> typing.Any:

        # print("[INFO] I am job ")
        method_to_call = getattr(target_obj, target_method)
        # time.sleep(input)
        result = method_to_call(**kwargs)
        return result

    @staticmethod
    def _log_configurer(log_queue: Queue):
        '''
        Configure logging handlers for a worker
        '''
        h = WorkerQueueHandler(log_queue)
        root = logging.getLogger()
        root.addHandler(h)

        # TODO: Now, sending all messages. Should set level based on logging level.
        # root.setLevel(logging.DEBUG)

    @staticmethod
    def _logger_thread(q: Queue):
        '''
        Thread on main process to wait for logging events
        '''
        while True:
            record = q.get()
            # print('log record:', record)
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)

    @staticmethod
    def _internal_worker_process(args: typing.Tuple[Queue, Queue, Queue, typing.Callable]) -> None:
        """
        The worker process iteratively checks the arguments_queue. It runs the target method with
        the arguments from top of arguments_queue. The worker finally pushes the results to the
        results queue for main process to read from it.
        Args:
            args: typing.Tuple[Queue, Queue, typing.Callable]

        """
        arguments_queue: Queue = args[0]
        result_queue: Queue = args[1]
        target: typing.Callable = args[2]
        log_queue: Queue = args[3]
        log_configurer: typing.Callable = args[4]

        # Configure logging
        log_configurer(log_queue)

        # _logger.debug("worker process started {}".format(current_process().name))
        # print(f"[INFO] {current_process().name} > worker process started")
        _logger.info("worker process started")
        _current_work_pids.append(os.getpid())
        counter: int = 0
        error_count: int = 0
        while True:
            if error_count > 3:
                break
            try:
                # wait until a new job is available
                # print(f"[INFO] {current_process().name} > waiting on new jobs")
                _logger.info("waiting on new jobs")
                kwargs = arguments_queue.get(block=True)
                _logger.info("copying")
                kwargs_copy = copy.copy(kwargs)
                # execute the job
                try:
                    # TODO add timelimit to single work in the worker
                    # print(f"[INFO] {current_process().name} > executing job")
                    result = target(**kwargs)
                    # assert hasattr(result['fitted_pipeline'], 'runtime'), \
                    #     '[DJM] Eval does not have runtime'
                except:
                    _logger.exception(
                        f'Target evaluation failed {hash(str(kwargs))}', exc_info=True)
                    # print(f'[INFO] {current_process().name} > Target evaluation failed {hash(str(kwargs))}')
                    # traceback.print_exc()
                    # _logger.error(traceback.format_exc())
                    result = None

                # push the results
                if result is not None:
                    result_simplified = result.copy()
                    if "ensemble_tunning_result" in result:
                        result_simplified.pop("ensemble_tunning_result")

                _logger.info(f"Pushing Results: {result['id'] if result and 'id' in result else 'NONE'}")
                _logger.debug(f"Pushing Results={result} kwargs={kwargs}")

                try:
                    result_queue.put((kwargs, result))
                except BrokenPipeError:
                    _logger.exception(f"Result queue put failed. Broken Pipe.")
                    exit(1)
                except:
                    # traceback.print_exc()
                    _logger.exception(f"Result queue put failed.", exc_info=True)
                    _logger.info(f"Result queue is full: {result_queue.full()}")

                    try:
                        _logger.info("Pushing result None. Maybe Result failed to pickle.")
                        result_queue.put((kwargs_copy, None))
                    except:
                        # traceback.print_exc()
                        # _logger.exception(f"{current_process().name} > {traceback.format_exc()}")
                        # print(f"[INFO] {current_process().name} > cannot even push None")
                        _logger.exception(f"Result queue put failed with empty Result.", exc_info=True)
                        _logger.info("Cannot even push None")
                        exit(1)

                    # exit(1)
                counter += 1
                # print(f"[INFO] {current_process().name} > is Idle, done {counter} jobs")
                _logger.info("is Idle, done {counter} jobs")
            except BrokenPipeError:
                error_count += 1
                print(f"{current_process().name:17} > Broken Pipe. Error count={error_count}")
                _logger.exception(f"Broken Pipe. Error count={error_count}")
            except Exception:
                error_count += 1
                print(f"{current_process().name:17} > Unexpected Exception. Error count={error_count}")
                _logger.exception(f"Unexpected Exception. Error count={error_count}", exc_info=True)
        print(f"{current_process().name:17} > Worker EXITING")
        _logger.warning('Worker EXITING')


    def push_job(self, kwargs_bundle: typing.Dict = {}) -> int:
        """
        The method creates a new process for the given job and returns the pid of the job
        Args:
            target:
            name:
            kwargs_bundle:

        Returns: int
            hash of the input argument

        """
        hint_message = "kwargs must be a dict with format: " \
                       "{\'target_obj\': ... , " \
                       "\'target_method\': ..., " \
                       "\'kwargs\': {[arg_name]: ...,}}"
        assert isinstance(kwargs_bundle, dict), hint_message
        assert all(
            l in kwargs_bundle for l in ['target_obj', 'target_method', 'kwargs']), hint_message
        assert isinstance(kwargs_bundle['kwargs'], dict), hint_message

        with self.argument_lock:
            self.ongoing_jobs += 1
            self.arguments_queue.put(kwargs_bundle)
        # self.result_queue_size = None

        return hash(str(kwargs_bundle))

    def pop_job(self, block: bool = False, timeout=None) -> typing.Tuple[typing.Dict, typing.Any]:
        """
        Pops the results from results queue
        Args:
            block: bool
            Is the pop blocking or non-blocking

        Returns:

        """
        _logger.info(f"# ongoing_jobs {self.ongoing_jobs}")
        print(f"# ongoing_jobs {self.ongoing_jobs}")

        with self.result_lock:
            (kwargs, results) = self.result_queue.get(block=block, timeout=timeout)
            self.ongoing_jobs -= 1
            print(f"[PID] pid:{os.getpid()}")

        # _logger.info(f"[INFO] end of pop # ongoing_jobs {self.ongoing_jobs}")
        return (kwargs, results)

        # self.result_queue_size = self.result_queue.qsize()

        # #!!!! error happened here
        # if self.result_queue_size > 0:
        #     _logger.debug("result_queue size is {}".format(str(self.result_queue.qsize())))
        #     (kwargs, results) = self.result_queue.get(block=block)
        #     self.ongoing_jobs -= 1
        #     print(f"[PID] pid:{os.getpid()}")
        #     self.Qlock.release()
        #     # _logger.info(f"[INFO] end of pop # ongoing_jobs {self.ongoing_jobs}")
        #     return (kwargs, results)
        # else:
        #     self.ongoing_jobs -= 1
        #     print(f"[PID] pid:{os.getpid()}")
        #     self.Qlock.release()
        #     return (None, None)

    def any_pending_job(self):
        return not self.arguments_queue.empty()

    def is_idle(self):
        return self.are_queues_empty() and self.are_workers_idle()

    def are_workers_idle(self):
        print(f"ongoing Jobs:{self.ongoing_jobs}")
        _logger.info(f"ongoing Jobs:{self.ongoing_jobs}")
        return self.ongoing_jobs == 0

    def are_queues_empty(self) -> bool:
        # _logger.info(f"arguments_queue:{len(self.arguments_queue)}, "
        #              f"result_queue:{len(self.result_queue)}")
        _logger.debug(f"are_queues_empty: {self.arguments_queue.empty()} and "
                      f"{self.result_queue.empty()}")
        return self.arguments_queue.empty() and self.result_queue.empty()

    def check_timeout(self):
        """
        Checks the timeout is reached.
        Returns:
            None
        Raises:
            TimeoutError: if the timeout is reached
        """
        elapsed_sec = ceil(time.perf_counter() - self.start_time)
        if elapsed_sec > self._timeout_sec:
            raise TimeoutError("Timeout reached: {}/{}".format(elapsed_sec, self.timeout_sec))

    def reset(self):
        '''
        Cancel timer and clear the job queue.
        '''
        self._timeout_sec = -1
        if self.timer:
            self.timer.cancel()
        self._clear_jobs()

    def kill_job_manager(self):
        """
        Safely kills the jobManager and all of its components
        Returns:
            None
        """
        _logger.warning('===DO YOU REALLY WANT TO KILL THE JOB MANAGER===')

        # Send sentinel to stop logging listener
        self.log_queue.put(None)

        _logger.debug("self.job_pool.terminate()")
        self.job_pool.terminate()

        _logger.debug("self.manager.shutdown()")
        self.manager.shutdown()

        # _logger.debug("kill_child_processes()")
        # DistributedJobManager.kill_child_processes()

        _logger.debug("self.kill_timer()")
        self.kill_timer()

    def kill_timer(self):
        if self.timer:
            _logger.warning(f"timer killed")
            self.timer.cancel()

    def _setup_timeout_timer(self):
        self.start_time = time.perf_counter()
        if self.timer_response == TimerResponse.KILL_WORKERS:
            self.timer = Timer(self._timeout_sec, self._kill_me)
        else:
            self.timer = Timer(self._timeout_sec, self._stop_worker_jobs)
        self.timer.start()
        _logger.warning(f"timer started: {self._timeout_sec/60} min")

    def _stop_worker_jobs(self):
        _logger.warning("search TIMEOUT reached! Stopping worker jobs. Actually just clearing the queue.")
        self._clear_jobs()

    def _clear_jobs(self):
        with self.argument_lock:
            _logger.info(f"Clearing {self.ongoing_jobs} jobs from queue")
            self.ongoing_jobs = 0
            while not self.arguments_queue.empty():
                self.arguments_queue.get()

    def _kill_me(self):
        _logger.warning("search TIMEOUT reached! Killing search Process")
        self.kill_job_manager()
        self.kill_timer()
        os._exit(0)
        # os.kill(os.getpid(), 9)

    @staticmethod
    def kill_child_processes():
        process_id = os.getpid()
        parent = psutil.Process(process_id)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
