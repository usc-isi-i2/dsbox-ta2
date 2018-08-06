import time
import typing
import os
from threading import Timer
from math import ceil
import traceback
from multiprocessing import Pool, Queue, Manager, Semaphore

class DistributedJobManager:
    def __init__(self, proc_num: int=4, timeout: int=55):

        self.start_time = time.perf_counter()
        self.proc_num = proc_num
        self.timeout = timeout

        self.manager = Manager()
        self.arguments_queue: Queue = self.manager.Queue()
        self.result_queue: Queue = self.manager.Queue()

        self.worker_sema: Semaphore = self.manager.Semaphore(value=proc_num)

        # initialize
        self.job_pool: Pool = None

        self._setup_timeout_timer()

        self.timer: Timer = None

    def start_workers(self, target: typing.Callable):
        self.job_pool = Pool(processes=self.proc_num)
        self.job_pool.map_async(
            func=DistributedJobManager._worker_process,
            iterable=
            [(self.arguments_queue, self.result_queue, target, self.worker_sema,)
             for a in range(self.proc_num)]
        )
        self.job_pool.close()  # prevents any additional worker to be added to the pool

    @staticmethod
    def _worker_process(args: typing.Tuple[Queue, Queue, typing.Callable, Semaphore]) -> None:
        """
        The worker process iteratively checks the arguments_queue. It runs the target method with
        the arguments from top of arguments_queue. The worker finally pushes the results to the
        results queue for main process to read from it.
        Args:
            args: typing.Tuple[Queue, Queue, typing.Callable, Semaphore]

        """
        arguments_queue: Queue = args[0]
        result_queue: Queue = args[1]
        target: typing.Callable = args[2]
        worker_sema: Semaphore = args[3]

        print("[INFO] worker process started")
        counter: int = 0
        while True:
            # wait until a new job is available
            kwargs = arguments_queue.get(block=True)
            # print("[INFO] Job {} got.".format(job_id))
            worker_sema.acquire(blocking=True)
            # execute the job
            try:
                # TODO add timelimit to single work in the worker
                result = target(**kwargs)
            except:
                traceback.print_exc()
                result = None

            # push the results
            print("[INFO] Pushing Results")
            result_queue.put((kwargs, result))
            worker_sema.release()
            # print("[INFO] Job {} done.".format(job_id))
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

        return (kwargs, results)

    def any_pending_job(self):
        return not self.arguments_queue.empty()

    def is_idle(self):
        return self.are_queues_empty() and self.are_workers_idle()

    def are_workers_idle(self):
        acquire_out = [self.worker_sema.acquire(blocking=False) for _ in range(self.proc_num)]
        are_workers_idle = True
        for b in acquire_out:
            if b:
                self.worker_sema.release()
            else:
                are_workers_idle = False
        return are_workers_idle

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
            raise TimeoutError("[INFO] Timeout reached: {}/{}".format(elapsed_min, self.timeout))

    def kill_job_mananger(self):
        self.job_pool.terminate()
        self.manager.shutdown()

    def _setup_timeout_timer(self):
        self.timer = Timer(self.timeout*60, self._kill_me)
        self.timer.start()
        print("[INFO] timer started: {} min".format(self.timeout))

    def _kill_me(self):
        print("[INFO] search TIMEOUT reached! Killing search Process")
        self.kill_job_mananger()
        os._exit(0)
        # os.kill(os.getpid(), 9)

    # def run_with_timeout(group=None, target: typing.Callable = None, name: str = None,
    #                      kwargs: typing.Dict = {},
    #                      timeout: float = None):
    #     m = Manager()
    #
    #
    #     p = Process(group=group,
    #                 target=lambda a: wrapperMethod(target, kwargs),
    #                 name=name,
    #                 args=args,
    #                 kwargs=kwargs)
    #     p.start()
    #     while p.is_alive():
    #         elapsed_min = ceil((time.perf_counter() - training_start) / 60)
    #         # print(STYLE + "[INFO] Elapsed:"+str(elapsed_min)+","+str(p.is_alive()))
    #         if elapsed_min >= Controller.TIMEOUT:
    #             self._logger.info(STYLE + "[WARN] Timeout reached!")
    #             print(STYLE + "[WARN] Timeout reached!")
    #             timeout = True
    #             p.terminate()
    #             break  # break the while loop
    #         time.sleep(1)