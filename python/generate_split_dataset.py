import os
import traceback
import copy
import subprocess
import json
import pandas as pd
from d3m.metadata.problem import parse_problem_description
from d3m.container.dataset import D3MDatasetLoader
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

def split_dataset(problem_info, dataset, data_type_cannot_split, task_type_can_split, random_state=42, test_size=0.2, n_splits=1, need_test_dataset = True):
    '''
        Split dataset into 2 parts for training and test
    '''
    import pdb
    pdb.set_trace()
    task_type = problem_info["task_type"]  #['problem']['task_type'].name  # 'classification' 'regression'
    res_id = problem_info["res_id"]
    target_index = problem_info["target_index"]
    data_type = problem_info["data_type"]


    train_return = []
    test_return = []

    cannot_split = False

    for each in data_type:
        if each in data_type_cannot_split:
            cannot_split = True
            break

    # check second time if the program think we still can split
    if not cannot_split:
        if task_type is not list:
            task_type_check = [task_type]

        for each in task_type_check:
            if each not in task_type_can_split:
                cannot_split = True
                break

    # if the dataset type in the list that we should not split
    if cannot_split:
        return None

    # if the dataset type can be split
    else:
        if task_type == 'CLASSIFICATION':
            try:
                # Use stratified sample to split the dataset
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                sss.get_n_splits(dataset[res_id], dataset[res_id].iloc[:, target_index])

                for train_index, test_index in sss.split(dataset[res_id], dataset[res_id].iloc[:, target_index]):
                    #
                    if need_test_dataset:
                        train = pd.DataFrame(index = train_index, columns = ['d3mIndex', 'type', 'repeat', 'fold'])
                        train['d3mIndex'] = train.index
                        train['type'] = 'TRAIN'
                        train['repeat'] = 0
                        train['fold'] = 0
                        test = pd.DataFrame(index = test_index, columns = ['d3mIndex', 'type', 'repeat', 'fold'])
                        test['d3mIndex'] = test.index
                        test['type'] = 'TEST'
                        test['repeat'] = 0
                        test['fold'] = 0
                        result = pd.concat([train, test], axis=0)
                        result = result.sort_index()
                    else:
                        # special condition for large dataset, we still need to keep the things inside of the dataset
                        indf = dataset[res_id]
                        outdf_train = pd.DataFrame(columns = dataset[res_id].columns)
                        for each_index in train_index:
                            each_slice = indf.loc[each_index]
                            #each_slice['d3mIndex'] = int(each_slice['d3mIndex'])
                            outdf_train = outdf_train.append(each_slice,ignore_index = True)
                        result = copy.copy(dataset)
                        result[res_id] = outdf_train
                    return result
            except:
                # Do not split stratified shuffle fails
                return None

        else:
            # Use random split
            try:
                ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
                ss.get_n_splits(dataset[res_id])
                for train_index, test_index in ss.split(dataset[res_id]):
                    if need_test_dataset:
                        train = pd.DataFrame(index = train_index, columns = ['d3mIndex', 'type', 'repeat', 'fold'])
                        train['d3mIndex'] = train.index
                        train['type'] = 'TRAIN'
                        train['repeat'] = 0
                        train['fold'] = 0
                        test = pd.DataFrame(index = test_index, columns = ['d3mIndex', 'type', 'repeat', 'fold'])
                        test['d3mIndex'] = test.index
                        test['type'] = 'TEST'
                        test['repeat'] = 0
                        test['fold'] = 0
                        result = pd.concat([train, test], axis=0)
                        result = result.sort_index()
                    else:
                        # special condition for large dataset, we still need to keep the things inside of the dataset
                        indf = dataset[res_id]
                        outdf_train = pd.DataFrame(columns = dataset[res_id].columns)
                        for each_index in train_index:
                            each_slice = indf.loc[each_index]
                            #each_slice['d3mIndex'] = int(each_slice['d3mIndex'])
                            outdf_train = outdf_train.append(each_slice,ignore_index = True)
                        result = copy.copy(dataset)
                        result[res_id] = outdf_train
                    return result
            except:
                # Do not split stratified shuffle fails
                return None

def list_files(base_directory, level = 1):
    '''
        find all folders of given path
    '''
    files = []
    if level >= 0:
        path = os.scandir(base_directory)
        for p in path:
            if p.is_dir():
            #     files.extend(list_files(p.path, level - 1))
            # else:
                files.append(os.path.relpath(p.path, start=base_directory))
    return files

def main() -> None:
    '''
        This program will genearate the splitted dataset (train/test) part base on our split function from Controller
        Then, based on the splitted dataset, run the d3m solution pipeline to get a new baseline so that we can compare our results
        more accurate

        Following parameters can be changed:
        1. dataset_path: (type = str) the path to target dataset
        2. replace: (type = Bool) determine whether to replace the split file or not (if False, will rename the old file)
        3. rund3mPipeline: (type = Bool) determine whether to run d3m's baseline pipeline or not
        4. wait (type = Bool) determine whether to let the function wait the one human pipeline finished and do next or not
        5. threshold_index_length: (type = int) the maximum allow size to process directly, here it is 100000 (same as our dsbox system now)
    '''
    dataset_path = "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/training_datasets/LL0/"
    dataset_path2 = copy.deepcopy(dataset_path)
    replace = False
    wait = True
    rund3mPipeline = True
    threshold_index_length = 100000
    #threshold_index_length = 100
    failed_dataset = {}
    files = list_files(base_directory = dataset_path)
    print("Totally",len(files), "datasets detected.")
    
    #!!! If you want only to test with given dataset, change here like the example
    #files = ["LL0_1_anneal"]
    
    # other info
    data_type_cannot_split = ["graph","edgeList", "audio"]
    task_type_can_split = ["CLASSIFICATION","REGRESSION","TIME_SERIES_FORECASTING"]

    for each_dataset in files:
        # 3 folder path may needed in the future
        print("------------------------------------------------------------------------------")
        print("Now processing dataset", each_dataset)
        problem_path = dataset_path2 + each_dataset + "/" + each_dataset + "_problem"
        dataset_path = dataset_path2 + each_dataset + "/" + each_dataset + "_dataset"
        solution_path = dataset_path2 + each_dataset + "/" + each_dataset + "_solution"
        split_file_loc = os.path.join(problem_path, "dataSplits.csv")
        split_file_loc_old = os.path.join(problem_path, "dataSplits_org.csv")
        problem_doc_loc = os.path.join(problem_path, "problemDoc.json")
        dataset_doc_loc = os.path.join(dataset_path, "datasetDoc.json")

        try:
            # get problem_info
            problem_info = {}
            problem = parse_problem_description(problem_doc_loc)
            taskSourceType = set()  # set the type to be set so that we can ignore the repeat elements
            with open(dataset_doc_loc, 'r') as dataset_description_file:
                dataset_description = json.load(dataset_description_file)
                for each_type in dataset_description["dataResources"]:
                    taskSourceType.add(each_type["resType"])
            problem_info["data_type"] = taskSourceType
            problem_info["task_type"] = problem['problem']['task_type'].name  # 'classification' 'regression'
            for i in range(len(problem['inputs'])):
                if 'targets' in problem['inputs'][i]:
                    break
            problem_info["res_id"] = problem['inputs'][i]['targets'][0]['resource_id']
            problem_info["target_index"] = []
            for each_target in problem['inputs'][i]['targets']:
                problem_info["target_index"].append(each_target["column_index"])

            # Dataset
            loader = D3MDatasetLoader()
            json_file = os.path.abspath(dataset_doc_loc)
            all_dataset_uri = 'file://{}'.format(json_file)
            all_dataset = loader.load(dataset_uri=all_dataset_uri)
            main_res_shape = all_dataset[problem_info["res_id"]].shape
            # special condition for dataset's index is very large
            if main_res_shape[0] > threshold_index_length:
                if main_res_shape[1] > 20:
                    threshold_index_length = int(threshold_index_length * 0.3)
                index_removed_percent = 1 - float(threshold_index_length) / float(main_res_shape[0])
                # ignore the test part
                all_dataset = split_dataset(dataset = all_dataset, 
                                            test_size = index_removed_percent, 
                                            need_test_dataset = False, 
                                            problem_info = problem_info, 
                                            data_type_cannot_split = data_type_cannot_split, 
                                            task_type_can_split = task_type_can_split)
        except:
            failed_dataset[each_dataset] = "Failed with incomplete dataset."
            continue
        # end collection dataset information

        # split dataset part
        split_result = split_dataset(dataset = all_dataset, 
                                     problem_info = problem_info, 
                                     data_type_cannot_split = data_type_cannot_split, 
                                     task_type_can_split = task_type_can_split)
        if split_result is None:
            failed_dataset[each_dataset] = "Failed with splitting unsuccessful."
            continue

        else:
            # rename the file if necessary
            if os.path.isfile(split_file_loc) and not replace:
                print("old split file found! Will rename it.")
                os.rename(split_file_loc, split_file_loc_old)
            # output the new split.csv file
            split_result.to_csv(split_file_loc)

        # Run the solution's pipeline (pipeline from d3m)
        if rund3mPipeline:
            try:
                cmd1 = 'cd ' + solution_path
                cmd2 = 'sh run.sh'
                cmd = cmd1 + " && " + cmd2
                #print()
                child = subprocess.Popen(cmd, shell=True)
                if wait:
                    child.wait()
            except:
                print("Running the human solution pipeline on dataset", each_dataset, "failed.")
                traceback.print_exc()
            # add the failed reason
            if child.returncode != 0:
                failed_dataset[each_dataset] = "Failed with running solution pipeline unsuccessful."
        print("----------------------------------------------------------------------")


    print("Program finished!")
    print("Totally",len(failed_dataset), "running failed:")
    if len(failed_dataset) > 0:
        for key, value in failed_dataset.items():
            print("  ", key.ljust(40), ":", value)


if __name__ == '__main__':
    main()