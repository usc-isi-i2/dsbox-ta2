import multiprocessing as mp
import time
import random
import psutil
import os
import logging
import json
import typing
from datetime import datetime
from bisect import bisect_left
from multiprocessing import Process, Manager
from collections import defaultdict
from dsbox.JobManager.DistributedJobManager import _current_work_pids

_logger = logging.getLogger(__name__)

class UsageMonitor:
    def __init__(self):
        m = Manager()
        self.recorder = m.list()
        self.target_pids = set()

    def add_target_process(self) -> None:
        for i in range(len(_current_work_pids)):
            self.target_pids.add(_current_work_pids.pop())
        _logger.info("Current totally " + str(len(self.target_pids)) + " pids are in target list.")
        _logger.info(str(self.target_pids))

    def start_recording_resource_usage(self):
        wait_time = 1
        frequency = 5
        need_record = logging.getLogger("dsbox.template.runtime").level <= 10
        if need_record:
            _logger.warning("Start recording the usage of the resources!")
            self.add_target_process()
            self.measure_process = Process(target=UsageMonitor.measure_usage_for_multiple_process, args=(self.recorder, self.target_pids,wait_time, frequency))
            self.measure_process.start() 
        else:
            _logger.warning("The logging level of dsbox.template.runtime is higher than debug, no resource usage will be recorded!")

    def stop_recording_resource_usage(self, file_loc):
        if self.measure_process is None:
            _logger.warning(ValueError("No exist recording process found!"))
        else:
            self.measure_process.terminate()
            _logger.info("Recording on resource ended!")
            self.write_usage_to_status(file_loc)

    def write_usage_to_status(self, file_loc):
        self.usage_record_dict = dict()
        self.usage_record_timeline = list()
        for _ in range(len(self.recorder)):
            each_record = self.recorder.pop()
            time = each_record[0]
            all_cpu_usage = each_record[1]
            all_memory_usage = each_record[2]
            self.usage_record_dict[time] = (all_cpu_usage, all_memory_usage)
            self.usage_record_timeline.append(time)
        self.usage_record_timeline.sort(reverse=False)

        usage_file_loc = os.path.join(file_loc, "pipelines_status")
        for root, _, files in os.walk(usage_file_loc):
            for filename in files:
                if (filename.startswith("fit") or filename.startswith("produce")) and filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    self.add_usage_to_file(file_path)

    def add_usage_to_file(self, file_path):
        with open(file_path, "r") as f:
            temp = json.load(f)
        current_pid = temp.get("pid")
        if not current_pid:
            _logger.error("No pid information found for record file " + file_path)
            return
        temp = self.add_record_in_one_step(temp, current_pid)
        for i, each_step in enumerate(temp["steps"]):
            temp["steps"][i] = self.add_record_in_one_step(temp["steps"][i], current_pid)
        with open(file_path, "w") as f:
            temp = json.dump(temp, f, separators=(',', ':'),indent=4)

    def add_record_in_one_step(self, record_part, current_pid):
        start_time = record_part["start"]
        end_time = record_part["end"]
        record_pos_start = bisect_left(self.usage_record_timeline, start_time)
        record_pos_end = bisect_left(self.usage_record_timeline, end_time)
        record_range = self.usage_record_timeline[record_pos_start: record_pos_end]
        if len(record_range) == 0:
            _logger.warning("No correspoing record found for {} at time range from {} to {}".format(str(current_pid), start_time, end_time))
            return record_part

        current_record_cpu_usage = dict()
        current_record_memory_usage = dict()
        try:
            for each_timestamp in record_range:
                # if type(self.usage_record_dict[each_timestamp][0][current_pid]) is not float or type(self.usage_record_dict[each_timestamp][1][current_pid]) is not float:
                #     import pdb
                #     pdb.set_trace()

                current_record_cpu_usage[each_timestamp] = self.usage_record_dict[each_timestamp][0][current_pid]
                current_record_memory_usage[each_timestamp] = self.usage_record_dict[each_timestamp][1][current_pid]
            record_part["resource_usage_cpu"] = current_record_cpu_usage
            record_part["resource_usage_memory"] = current_record_memory_usage
            record_part["resource_usage_cpu_average"] = sum(current_record_cpu_usage.values()) / len(current_record_cpu_usage.values())
            record_part["resource_usage_memory_average"] = sum(current_record_memory_usage.values()) / len(current_record_memory_usage.values())
        except:
            pass
        return record_part

    @staticmethod
    def get_all_subprocesses(target_pids) -> dict:
        """
        Function used to get all child processes from given target pids
        Returns a dict, the key is the all found processes and the value is the parent processes given from target_pids
        """
        target_pids_dict = dict()
        for each_pid in target_pids:
            target_pids_dict[each_pid] = each_pid

        # get child processes
        all_processes = list(psutil.process_iter())
        for proc in all_processes:
            if proc.pid in target_pids:
                current_parent_pid = proc.pid
                target_process = proc
                for child in target_process.children(recursive=True):
                    target_pids_dict[child.pid] = current_parent_pid
        return target_pids_dict

    @staticmethod
    def measure_usage_for_multiple_process(recorder, target_pids, wait_time_get_data=1, frequency_update_pids=5):
        """
        Function used to measure the usage of given processes, this can only work for non-daemon processes
        """
        # print("target pid is:", target_pid)
        # print("current pid is:", str(os.getpid()))
        # time.sleep(1)
        while True:
            target_pids_dict = UsageMonitor.get_all_subprocesses(target_pids)
            _logger.debug("Following is the monitor list")
            _logger.debug(str(target_pids_dict))

            for i in range(frequency_update_pids):
                result = UsageMonitor.get_all_usage(target_pids_dict)
                recorder.append(result)
                time.sleep(wait_time_get_data)


    @staticmethod
    def get_all_usage(target_pids_dict):# -> typing.Tuple(str, dict, dict):
        """
        """
        all_processes = list(psutil.process_iter()) 
        all_memory_usage = defaultdict(float)
        all_cpu_usage = defaultdict(float)
        for proc in all_processes:
            # use try to prevent process died and cause program crashed
            try:
                if proc.pid in target_pids_dict.keys():
                    # add the target process's usage to its parent usage count
                    cpu_usage = proc.cpu_percent() # in percentage
                    memory_usage = proc.memory_info().rss / 1024 / 1024 # in MB
                    all_memory_usage[target_pids_dict[proc.pid]] += memory_usage
                    all_cpu_usage[target_pids_dict[proc.pid]] += cpu_usage
            except:
                pass
        stamp = datetime.utcfromtimestamp(time.time())
        stamp = stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        one_time_record = [stamp, all_cpu_usage, all_memory_usage]
        return one_time_record

    @staticmethod
    def measure_usage_for_one_process(recorder, target_pid, frequency=1):
        """
        Function used to measure the usage of given processes, this can only work for non-daemon processes
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
                # 
                except:
                    pass
            stamp = str(datetime. now())
            recorder.append([stamp, all_cpu_percent, all_memory_usage])
