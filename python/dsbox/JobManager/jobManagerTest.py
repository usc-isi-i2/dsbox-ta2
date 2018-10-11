import time
import random
from DistributedJobManager import DistributedJobManager


class bar():
    def __init__(self, a: int):
        self.a = a

    def work(self):
        sl = random.randint(0, 5)
        print(f"[INFO] sleeping for {sl}")
        time.sleep(sl)
        print(f"[INFO] I am working on {self.a}")
        return self.a



class foo():
    def __init__(self):
        self.m = DistributedJobManager(proc_num=4, timeout=10)
        self.b_list = [bar(random.random()) for _ in range(6)]

        # self.m._start_workers(foo.job_process)

    # @staticmethod
    # def job_process(in_val: bar, method: str):
    #     print("[INFO] I am job ", in_val)
    #     method_to_call = getattr(in_val, method)
    #     # time.sleep(input)
    #     print(method_to_call)
    #     result = method_to_call()
    #     return result

    def run(self):
        for i in range(10):
            num = random.choice(self.b_list)
            jid = self.m.push_job(kwargs={'target_obj': num, 'target_method': 'work'})
            print("[INFO] Job pushed ", (jid, num))

        time.sleep(2)

        while not self.m.is_idle():
            (num, result) = self.m.pop_job(block=True)
            print(f"[INFO] Job popped {num['target_obj']} -> {result}")
        # hash(str(num))
        print("[INFO] Killing job manager")
        self.m.kill_job_mananger()
        print("[INFO] Done")

        self.m.kill_timer()



o=foo()
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
