import multiprocessing as mp
from multiprocessing import Process
import time
import random
import psutil,time
import os

def measure_usage(recorder, target_pid, frequency=1):
    """
    Function used to measure the usage of current process
    """
    # print("target pid is:", target_pid)
    # print("current pid is:", str(os.getpid()))
    # time.sleep(1)
    target_pids = set()
    target_pids.add(target_pid)
    # get child processes
    for proc in psutil.process_iter():
        if proc.pid == target_pid:
            target_process = proc
            for child in target_process.children(recursive=True):
                target_pids.add(child.pid)
    # print("target pids are:", target_pids)
    # measure until the proces finished
    while True:
        all_processes = list(psutil.process_iter()) 
        time.sleep(frequency)
        all_memory_usage = 0
        all_cpu_percent = 0
        for proc in all_processes:
            try:
                if proc.pid in target_pids:
                    # process_name = proc.name()
                    # command = " " .join(proc.cmdline())
                    # print("parent is " + str(proc.parent()))
                    # print("children is " + str(proc.children()))
                    cpu_usage = proc.cpu_percent() # in percentage
                    memory_usage = proc.memory_info().rss / 1024 / 1024 # in MB
                    all_memory_usage += memory_usage
                    all_cpu_percent += cpu_usage
                        # print("child pid = ", child.pid)
                        # cpu_usage = child.cpu_percent() # in percentage
                        # memory_usage = child.memory_info().rss / 1024 / 1024 # in MB
                        # print("child " + str(child.pid) + " " + str(cpu_usage), memory_usage)
                        # all_memory_usage += memory_usage
                        # all_cpu_percent += cpu_usage
            # stamp = time.time()
            except:
                pass
        recorder.put([all_cpu_percent, all_memory_usage])


def run_one_full():
    while True:
        a = 1

if __name__ == "__main__":
    recorder = mp.Queue()
    frequency = 1
    # p.join()
    print("main_start!")
    p2 = Process(target=run_one_full)
    p2.start()
    time.sleep(0.1)
    p3 = Process(target=run_one_full)
    p3.start()
    time.sleep(0.1)
    p4 = Process(target=run_one_full)
    p4.start()
    time.sleep(0.1)

    p = Process(target=measure_usage, args=(recorder, os.getpid(),frequency))
    p.start() 

    time.sleep(5)
    # for i in range(10):
    while not recorder.empty():
        print(recorder.get())
        # time.sleep(frequency)

    p.terminate()
    p2.terminate()
    p3.terminate()
    p4.terminate()

    for i in range(1,10):
        print("-main!!!--%d---"%i)
        time.sleep(3)