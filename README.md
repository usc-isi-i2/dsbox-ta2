# dsbox-ta2
The DSBox TA2 component

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
4.	After replacing these things, go back to common-primitives folder and run "pip install -e ." to reinstall the common-primitive module.
5.	Then, you can run "$ python -m d3m.index search" to check whether the "denormalize" primitive appeared in the search results.
6.	Block the "dataset_split" function at dsbox/controller/controller.py for now (I have done this in this branch).
7.	Depending on the model you will use, you may need to update some of our dsbox modules. (e.g. if you are trying to run the sample template for the timeseries data in the dsbox-ta2 template)

