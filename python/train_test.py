import os
import subprocess
import sys

from multiprocessing.pool import ThreadPool
from pathlib import Path


def call_ta2search(command):
    print(command)
    print('\n')

    p = subprocess.Popen(command, shell=True)
    p.communicate()


num = 5
tp = ThreadPool(num)

home = str(Path.home())
# config_dir = sys.argv[2]
config_dir = home + "/dsbox/runs2/config-ll0/"

for conf in os.listdir(config_dir):
    # command = "python3 ta1-run-single-template --template " + sys.argv[1] + " " + os.path.join(config_dir, conf, 'search_config.json')
    command = "python ta2-search " + config_dir + conf

    tp.apply_async(call_ta2search, (command,))

tp.close()
tp.join()
