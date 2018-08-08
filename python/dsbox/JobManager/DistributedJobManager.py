import time
import typing
import os
import logging
from threading import Timer
from math import ceil
import traceback
from multiprocessing import Pool, Queue, Manager, Semaphore, current_process
from multiprocessing import get_logger

_logger = logging.getLogger(__name__)

class DistributedJobManager:
    def __init__(self, proc_num: int=4, timeout: int=55):

        self.start_time = time.perf_counter()
        self.proc_num = proc_num
        self.timeout = timeout

        self.manager = Manager()
        self.arguments_queue: Queue = self.manager.Queue()
        self.result_queue: Queue = self.manager.Queue()

        # initialize
        self.job_pool: Pool = None

        self._setup_timeout_timer()

        self.timer: Timer = None

        # status counter
        self.ongoing_jobs: int = 0

    def start_workers(self, target: typing.Callable):
        self.job_pool = Pool(processes=self.proc_num)
        self.job_pool.map_async(
            func=DistributedJobManager._worker_process,
            iterable=
            [(self.arguments_queue, self.result_queue, target,)
             for a in range(self.proc_num)]
        )
        self.job_pool.close()  # prevents any additional worker to be added to the pool

    @staticmethod
    def _worker_process(args: typing.Tuple[Queue, Queue, typing.Callable]) -> None:
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

        _logger.debug("worker process started {}".format(current_process()))
        counter: int = 0
        while True:
            # wait until a new job is available
            kwargs = arguments_queue.get(block=True)

            # execute the job
            try:
                # TODO add timelimit to single work in the worker
                result = target(**kwargs)
            except:
                _logger.exception('Evaulate pipeline failed')
                traceback.print_exc()
                result = None

            # push the results
            print("Pushing Results {}".format(current_process()))
            result_queue.put((kwargs, result))
            print("Worker is Idle {}".format(current_process()))

            counter += 1

    def push_job(self, kwargs: typing.Dict = {}) -> int:
        """
        The method creates a new process for the given job and returns the pid of the job
        Args:
            target:
            name:
            kwargs:

        Returns: int
            hash of the input argument

        """
        self.ongoing_jobs += 1
        self.arguments_queue.put(kwargs)

        return hash(str(kwargs))

    def pop_job(self, block: bool = False) -> typing.Tuple[typing.Dict, typing.Any]:
        """
        Pops the results from results queue
        Args:
            block: bool
            Is the pop blocking or non-blocking

        Returns:

        """

        (kwargs, results) = self.result_queue.get(block=block)
        self.ongoing_jobs -= 1
        return (kwargs, results)

    def any_pending_job(self):
        return not self.arguments_queue.empty()

    def is_idle(self):
        return self.are_queues_empty() and self.are_workers_idle()

    def are_workers_idle(self):
        print("ongoing Jobs:", self.ongoing_jobs)
        return self.ongoing_jobs == 0

    def are_queues_empty(self) -> bool:
        return self.arguments_queue.empty() and self.result_queue.empty()

    def check_timeout(self):
        """
        Checks the timeout is reached.
        Returns:
            None
        Raises:
            TimeoutError: if the timeout is reached
        """
        elapsed_min = ceil((time.perf_counter() - self.start_time) / 60)
        if elapsed_min > self.timeout:
            raise TimeoutError("Timeout reached: {}/{}".format(elapsed_min, self.timeout))

    def kill_job_mananger(self):
        self.job_pool.terminate()
        self.manager.shutdown()

    def _setup_timeout_timer(self):
        self.timer = Timer(self.timeout*60, self._kill_me)
        self.timer.start()
        print("timer started: {} min".format(self.timeout))

    def _kill_me(self):
        _logger.warning("search TIMEOUT reached! Killing search Process")
        self.kill_job_mananger()
        os._exit(0)
        # os.kill(os.getpid(), 9)
