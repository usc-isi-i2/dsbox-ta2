import sys
import json
import os

####################################################################################################################
# Entry point for the TA2 Evaluation system. This script will be aliased to the command ta2_search in the docker
# image and called when the container is run:
#
#    docker run -i --entrypoint /bin/bash <container_id> -c 'ta2_search $CONFIG_JSON_PATH'
#
# The $CONFIG_JSON_PATH contains the path to the problem and data as well as the output path
#
# See the following for more details:
#
#    https://datadrivendiscovery.org/wiki/pages/viewpage.action?spaceKey=gov&title=TA2+Submission+Guide
####################################################################################################################

## Grab the input parameters
print "main.py invoked to perform", sys.argv[1]
print "main.py invoked with config file", sys.argv[2]

## Parse the input JSON config file
with open(sys.argv[2]) as data_file:
    data = json.load(data_file)

problem = data["problem_schema"]
output = data["executables_root"]

## Call the ISI TA2 System with the provided parameters
# python run.py -p ../data/o_38 -l library -o outputs/o_38
os.system("python run.py -p " + problem + " -l library -o " + output)

