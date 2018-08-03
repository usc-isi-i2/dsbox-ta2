import time
import random
from DistributedJobManager import DistributedJobManager


m = DistributedJobManager(proc_num=4, timeout=10)


def job_process(input: int):
    print("[INFO] I am job ", input)
    # time.sleep(input)
    return input+10

m.start_workers(job_process)

for i in range(10):
    num = random.randrange(0, 10, 1)
    jid = m.push_job(kwargs={'input': num})
    print("[INFO] Job pushed ", (jid, num))

time.sleep(2)

while not m.is_idle():
    (num, result) = m.pop_job(block=True)
    print("[INFO] Job popped ", hash(str(num)))

m.kill_job_mananger()

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
