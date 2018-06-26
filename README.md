# dsbox-ta2
*** You need to do following things to ensure the new api system can run in your computer ***

Current version for new api:
1.	Update d3m module to "master" branch version
2.	Update common-primitives to "master" branch version
3.	Update sklearn-wrap to "dev-dist" branch version
4.	Update dsbox-ta2 to "new_test" branch version

Change to fit running on new api for now 
(you can find these changed file at our dsbox-ta2/python/d3m_modify folder):
1.	Replace the "base.py" and "pipeline.py" part at d3m/metadata
2.	Replace the " entry_points.ini" part at common_primitives
3.	Replace the "utils.py" and "denormalize.py" part at common_primitives/common_primitives
3.  Go back to the folder of common_primitives and run "pip install -e ." to reinstall the common_primitives module.

4.	Block the "dataset_split" function at dsbox/controller/controller.py for now (I have done this in this branch).
5.	Depending on the model you will use, you may need to update some of our dsbox modules. (e.g. if you are trying to run the sample template for the timeseries data in the dsbox-ta2 template)

If you successfully reinstall the package, you can run the following script to check whether the system have found the denormalize module.
This command will show all primitives found in the system:
$ python -m d3m.index search
