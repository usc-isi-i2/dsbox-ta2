import os
from pathlib import Path

import subprocess

home = str(Path.home())
config_dir = home + "/dsbox/runs2/config-seed/"

for conf in os.listdir(config_dir):
	command = "python ta2-search " + config_dir + conf
	print(command)
	subprocess.Popen(command, shell=True)
