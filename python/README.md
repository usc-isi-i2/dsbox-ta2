How to run the Planner
=======

python run.py -p [path to problem directory] -l [path to library directory] -o [output directory]
 
Example: python run.py -p ../data/o_38 -l library -o outputs/o_38

This will create the following:
outputs
└── o_38
    ├── executables
    │   ├── 1.184ef4e3-793c-11e7-9145-f40f243a35c2.py
    │   ├── 2.1852a235-793c-11e7-8170-f40f243a35c2.py
    │   ├── 3.18532117-793c-11e7-927d-f40f243a35c2.py
    │   ├── 4.1853727d-793c-11e7-8002-f40f243a35c2.py
    │   ├── ...
    ├── log.txt
    ├── models
    │   ├── 1.184ef4e3-793c-11e7-9145-f40f243a35c2.pkl
    │   ├── 2.1852a235-793c-11e7-8170-f40f243a35c2.pkl
    │   ├── 3.18532117-793c-11e7-927d-f40f243a35c2.pkl
    │   ├── 4.1853727d-793c-11e7-8002-f40f243a35c2.pkl
    │   └── ...
    └── pipelines.txt

The file "pipelines.txt" has a list of pipeline details ranked by metric values

The python files in the executable folders can be executed directly. The executable
file depends upon the respective models in the models directly.
Example: python outputs/o_38/executables/1.184ef4e3-793c-11e7-9145-f40f243a35c2.py

The executables and models are numbered according to their ranking.

The file "log.txt" contains the planner's logging information.
