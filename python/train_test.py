import os
import subprocess

from multiprocessing.pool import ThreadPool
from pathlib import Path

def call_ta2search(command):
	p = subprocess.Popen(command, shell=True)
	p.communicate()

num = 5
tp = ThreadPool(num)

home = str(Path.home())
config_dir = home + "/dsbox/runs2/config-ll0/"

for conf in os.listdir(config_dir):
	command = "python ta2-search " + config_dir + conf

	print(command)

	tp.apply_async(call_ta2search, (command,))

tp.close()
tp.join()
