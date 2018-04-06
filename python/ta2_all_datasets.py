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
    prnt = True
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
            #print(config)
            # don't change timeout, cpus, ram?

            # Start the controller
            
            controller = Controller(LIB_DIRECTORY)
            controller.initialize_from_config(config)
            controller.load_problem()
            # Setup a signal handler to exit gracefully
            # Either on an interrupt or after a certain time
            def write_results_and_exit(signal, frame):
                controller.write_training_results()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, write_results_and_exit)
            signal.signal(signal.SIGTERM, write_results_and_exit)
            signal.signal(signal.SIGALRM, write_results_and_exit)
            signal.alarm(TIMEOUT)
            # Load in data
            controller.initialize_training_data_from_config()

            # Start training
            controller.initialize_planners()

            for result in controller.train(PlannerEventHandler()):
                print('training ', result)
                if result == False:
                    print("ProblemNotImplemented")
                    sys.exit(148)
                pass


if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    sys.exit(main())
    #main()
