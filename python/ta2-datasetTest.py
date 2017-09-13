#!/usr/bin/env python

"""
Tests for running TA2 (using run.py, which is assumed to be on the same folder as this script.)
Expected input: the file directory containing the datasets. 
This program does not detail exceptions
"""


import sys
import run
import os
from os import listdir
from os.path import isfile,join
#Note: you need to install the mock package
from mock import patch

def main(argv=None): 
    #Load data folder with all the dsbox datasets to test (argv[1])
    #Quick test, should refine sys.argv
    dataDirectory = sys.argv[1]
    for f in listdir(dataDirectory):
        if not isfile(join(dataDirectory, f)):#For all directories
            print (f)
            testargs = ["run.py","-p", join(dataDirectory, f), "-l", "library","-o", "outputs"+os.sep+f]
            with patch.object(sys, 'argv', testargs):
                #print(sys.argv, len(sys.argv)) 
                print("Testing :"+f+"...")
                try:
                    run.main()#We invoke the run script with all the datasets.
                except:
                    print("Error while executing the datasets, see logs for more information")
    
if __name__ == "__main__":
    sys.exit(main())
