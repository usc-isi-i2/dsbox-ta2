import os
from pathlib import Path

import subprocess

home = str(Path.home())
config_dir = home + "/dsbox/runs2/config-ll0/"

for conf in os.listdir(config_dir):
	command = "python ta2-search " + config_dir + conf
	print(command)
	subprocess.Popen(command, shell=True)


# TODO: add rerun utility for nonexistent text files

# TODO: do a more elegant subprocess management - one process finishes start new process