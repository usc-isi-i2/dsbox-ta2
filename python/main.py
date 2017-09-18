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

## Grab the input parameters passed in on the docker command and display them
print "main.py invoked to perform", sys.argv[1]
print "main.py invoked with config file", sys.argv[2]

## Parse the input JSON config file
with open(sys.argv[2]) as data_file:
    data = json.load(data_file)

## Convert the provided parameters into the form that DSBox wants
problem = data["problem_schema"][ : data["problem_schema"].find("problemSchema.json")]
output = data["executables_root"][ : data["executables_root"].find("executables")]

print "problem directory is: ", problem
print "output directory is: ", output

## TODO: Chnage the way DSBox uses paths and file names to match the provided government format - this will prevent failures in case they make changes
## Here is a sample config.json:
# {
#     "problem_schema": "/baseball/problemSchema.json",
#     "dataset_schema": "/baseball/data/dataSchema.json",
#     "training_data_root": "/baseball/data",
#     "pipeline_logs_root": "/outputs/logs",
#     "executables_root": "/outputs/executables",
#     "temp_storage_root": "/temp"
# }


## Call the ISI TA2 System with the provided parameters
os.system("python run.py -p " + problem + " -l library -o " + output)

