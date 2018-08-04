import time
import typing
from math import ceil
import traceback
from multiprocessing import Pool, Queue, Manager

class DistributedJobManager:
    def __init__(self, proc_num: int=4, timeout: int=55):

        self.start_time = time.perf_counter()
        self.proc_num = proc_num
        self.timeout = timeout

        self.manager = Manager()
        self.arguments_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()

        # initialize
        self.job_pool: Pool = None

    def start_workers(self, target: typing.Callable):
        self.job_pool = Pool(processes=self.proc_num)
        self.job_pool.map_async(
            func=DistributedJobManager._worker_process,
            iterable=
            [(self.arguments_queue, self.result_queue, target,) for a in range(self.proc_num)]
        )
        self.job_pool.close()  # prevents any additional worker to be added to the pool

    @staticmethod
    def _worker_process(args: typing.Tuple[Queue, Queue, typing.Callable]) -> None:
        """
        The worker process iteratively checks the arguments_queue. It runs the target method with
        the arguments from top of arguments_queue. The worker finally pushes the results to the
        results queue for main process to read from it.
        Args:
            args: typing.Tuple[typing.Callable, Queue, Queue]

        """
        arguments_queue: Queue = args[0]
        result_queue: Queue = args[1]
        target: typing.Callable = args[2]

        print("[INFO] worker process started")
        counter: int = 0
        while True:
            # wait until a new job is available
            kwargs = arguments_queue.get(block=True)
            # print("[INFO] Job {} got.".format(job_id))
            # execute the job
            try:
                # TODO add timelimit to single work in the worker
                result = target(**kwargs)
            except:
                traceback.print_exc()
                result = None

            # push the results
            result_queue.put((kwargs, result))
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
        self.manager.shutdown()
        self.job_pool.terminate()


    # @staticmethod
    # def run_with_timeout(group=None, target: typing.Callable = None, name: str = None,
    #                      kwargs: typing.Dict = {},
    #                      timeout: float = None):
    #     m = Manager()
    #
    #     def wrapperMethod(func, output_cache, kwargs):
    #         # out_list = kwargs['process_output']
    #         output_cache.append(func(**kwargs))
    #
    #     # assert "process_output" not in kwargs, "Argument \'process_output\' is reserved
    # keyword."
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