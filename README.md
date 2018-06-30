# dsbox-ta2
The DSBox TA2 component
---------------------------------------------2018.6.29---------------------------------------------
New updates:
1. Now you can run the controller.test() function after running train and saved pipeline successfully.
2. To properly save the trained pipeline and run the TRAIN and TEST dataset correctly on the system, some modifications have been made on the config file loaded to ta2-search.
3. After each run of train(), the system would generate a json file which describe the pipelines you just run with its pipeline id.
4. After each run of train(), the system would generate an  folder with a folder named with pipeline id in the given directory's "excutatable" folder, these are the pickled primitives files.
5. After each run of test(), the system would generate a csv file which is the predictions results of the test dataset. For this test() function, the system will trying to find the given
6. Now the ta2-search will run Controller.train() and Controller.test() one time each, you can block one if you want.
7. The test function only tested with current template in "library.py" with dataset No.38, 22(this dataset's dataset_schema has some problem, at least for 3.0.0 version), 1491, 66 and 49.
8. Further addede template may not be able to save(depending on the format of the output of the pipeline), please let me know (may at slack @Jun Liu) if something can't work.
9. Now the validation Accurancy will always be 0 because we no longer have the true results for test data.

Here is one example new config input file.
{
"cpus": "10",
"dataset_schema": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/38_sick_dataset/datasetDoc.json",
"executables_root": "/Users/minazuki/Desktop/studies/master/2018Summer/data/executables",
"pipeline_logs_root": "/Users/minazuki/Desktop/studies/master/2018Summer/data/logs",
"problem_root": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/TRAIN/problem_TRAIN",
"problem_schema": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/TRAIN/problem_TRAIN/problemDoc.json",
"ram": "10Gi",
"temp_storage_root": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick_new/temp",
"timeout": 49,
"train_data_schema": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/TRAIN/dataset_TRAIN/datasetDoc.json",
"test_data_schema": "/Users/minazuki/Desktop/studies/master/2018Summer/data/datasets/seed_datasets_current/38_sick/TEST/dataset_TEST/datasetDoc.json",
"saving_folder_loc": "/Users/minazuki/Desktop/studies/master/2018Summer/data/saving_part",
"saved_pipeline_ID": ""
}
   
   new attributes for the config files:
   1. You should specifiy the "train_data_schema" and "test_data_schema" to the config so that the system can load the train and test dataset separately and make predictions on test dataset.
   2. An extra value "saving_folder_loc" have been added, this is used to specify where to save the pipeline-related files.
   3. An extra value "saved_pipeline_ID" have been added, this is used to specify which saved pipeline you want to use for test. If the value is blank like "", the system will try to load the latest modified pipeline for testing.
   
   TODO:
   Add the metadata.yml for the whole system.
   Example from https://gitlab.datadrivendiscovery.org/nist/submission/blob/master/example_ta2_submission/metadata.yml

---------------------------------------------2018.6.26---------------------------------------------
Plase also check the dsbox-experimentation instructions to do the installations with this new version API.
Current version for new api:
1.	Update d3m module to "master" branch version
2.	Update common-primitives to "master" branch version
3.	Update sklearn-wrap to "dist" branch version
4.	Update dsbox-ta2 to "template-2018-june" branch version
* You may need to run "pip install -e ." for each module after update

Change to fit running on new api for now 
(you can find these changed file at our dsbox-ta2/python/d3m_modify folder):
1.	Replace the "base.py" and "pipeline.py" part at d3m/metadata
2.	Replace the " entry_points.ini" part at common_primitives
3.	Replace the "denormalize.py" part at common_primitives/common_primitives
4.	After replacing these things, go back to common-primitives folder and run "pip install -e ." to reinstall the common-primitive module.
5.	Then, you can run "$ python -m d3m.index search" to check whether the "denormalize" primitive appeared in the search results.
6.	Block the "dataset_split" function at dsbox/controller/controller.py for now (I have done this in this branch).
7.	Depending on the model you will use, you may need to update some of our dsbox modules. (e.g. if you are trying to run the sample template for the timeseries data in the dsbox-ta2 template)

