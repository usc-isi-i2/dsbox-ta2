import time
import random
from DistributedJobManager import DistributedJobManager




class foo():
    def __init__(self):
        self.m = DistributedJobManager(proc_num=4, timeout=10)
        self.a = 10

        self.m.start_workers(foo.job_process)

    @staticmethod
    def job_process(in_val: int, a: int=10):
        print("[INFO] I am job ", in_val)
        # time.sleep(input)
        return in_val+a

    def run(self):
        for i in range(10):
            num = random.randrange(0, 10, 1)
            jid = self.m.push_job(kwargs={'in_val': num})
            print("[INFO] Job pushed ", (jid, num))

        time.sleep(2)

        while not self.m.is_idle():
            (num, result) = self.m.pop_job(block=True)
            print("[INFO] Job popped ", hash(str(num)))

        self.m.kill_job_mananger()



o = foo()
o.run()
# import multiprocessing, time
#
#
# def worker(args):
#     q = args[0]
#     p = args[1]
#     t = args[2]
#     while True:
#         a = q.get(block=True)
#         print("[INFO] got ", a)
#         p.put(t(a))
#
# def task(i):
#     return i+10
#
# def main():
#     manager = multiprocessing.Manager()
#     q = manager.Queue()
#     p = manager.Queue()
#     pool = multiprocessing.Pool()
#     pool.map_async(worker, [(q, p, task) for x in range(10)])
#     # pool.map_async()
#     for i in range(10):
#         q.put(i)
#
#     time.sleep(1)
#     while not (p.empty() and p.empty()):
#         print("[INFO] master got ", p.get())
#     pool.terminate()
#
#
# main()
