'''
    Run the system on training + test datasets.
'''

import os
import subprocess
import sys

from pathlib import Path

home = str(Path.home())

for conf in os.listdir(home + "/dsbox/runs2/config-seed/"):
    print("Working for", conf)
    os.system("python ta2-search /nas/home/stan/dsbox/runs2/config-seed/" + conf)
    print("\n" * 10)
