from dsbox_dev_setup import path_setup
path_setup()

import sys
import os
import json
import signal
from dsbox.planner.controller import Controller, Feature
from dsbox.planner.event_handler import PlannerEventHandler

TIMEOUT = 25*60 # Timeout after 25 minutes

DEBUG = 0
LIB_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/library"

DATA_DIR = '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data'

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    #program_name = os.path.basename(sys.argv[0])
    #program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    #program_usage = '''%s
#USAGE
#ta2-search <training_data_root_directory> <output_dir>
#''' % program_shortdesc

    if len(sys.argv) < 3 or sys.argv[0] == '-h':
        def_path = '/nas/home/kyao/dsbox/data/datadrivendiscovery.org/data/training_datasets/LL0/'
        print('def path')
        if os.path.exists(def_path): 
            dataset_folder = def_path
            output_dir = 'outputs/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            print('Exit')
            print(program_usage)
            exit(1)
    else:
        dataset_folder = sys.argv[1]
        output_dir = sys.argv[2]
    
    try:
        print(sys.argv[0])
        print(sys.argv[1])
        print(sys.argv[2])
    except:
        pass

    #conf_file = sys.argv[1]
    verbose = True
    cnt = 0

    for folder, sub_dirs, _ in os.walk(dataset_folder):
        dataset = folder.split('/')[-1]
        if folder != dataset_folder and '.git' not in folder:
            config = {}
            del sub_dirs[:]
            config["problem_schema"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_problem'), 'problemDoc.json')
            config["problem_root"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_problem'))
            config["dataset_schema"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_dataset'), 'datasetDoc.json')
            config["training_data_root"] = os.path.join(dataset_folder, str(dataset), str(dataset+'_dataset'))
            config["pipeline_logs_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'logs')
            config["executables_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'executables')  
            config["temp_storage_root"] =  os.path.join(os.getcwd(), output_dir, str(dataset),'temp')
            config["timeout"] = TIMEOUT
            config["cpus"] = "4"
            config["ram"] = "4Gi"
            config["include"] = ["d3m.primitives.sklearn_wrap.SKAdaBoostClassifier", \
        "d3m.primitives.sklearn_wrap.SKExtraTreesClassifier", \
        "d3m.primitives.sklearn_wrap.SKLogisticRegression", \
        "d3m.primitives.sklearn_wrap.SKLinearSVC", \
        "d3m.primitives.sklearn_wrap.SKKNeighborsClassifier", \
        "d3m.primitives.sklearn_wrap.SKLinearDiscriminantAnalysis", \
        "d3m.primitives.sklearn_wrap.SKRandomForestClassifier", \
        "d3m.primitives.sklearn_wrap.SKGaussianNB", \
        "d3m.primitives.sklearn_wrap.SKMultinomialNB", \
        "d3m.primitives.sklearn_wrap.SKSGDClassifier", \
        "d3m.primitives.sklearn_wrap.SKQuadraticDiscriminantAnalysis"]
#            config["include"] = ["d3m.primitives.sklearn_wrap.SKAdaBoostClassifier", \ 
#                                "LINEAR_DISCRIMINANT_ANALYSIS", \ 
#                                "NAIVE_BAYES_CLASSIFIER"]
            config["exclude"] = ["*"]
#            config["exclude"] = ["d3m.primitives.sklearn_wrap.SKMultinomialNB", \
#                                "d3m.primitives.sklearn_wrap.SKBernoulliNB"]
            #print(config)

            conf_dir = "conf"
            with open(conf_dir + "/" + dataset + ".conf", "w+") as fp:
                json.dump(config, fp, sort_keys=False, indent=4)
#            break

        cnt = cnt + 1
        print(str(cnt) + ". " + "Generated config file for", dataset)


if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    sys.exit(main())
    #main()
