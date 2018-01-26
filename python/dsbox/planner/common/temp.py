'''
Created on Jan 25, 2018

@author: ktyao
'''

PRIMITIVES_REPO = ''
if not PRIMITIVES_REPO:
    print('set PRIMITIVES_REPO to clone of rep at https://gitlab.datadrivendiscovery.org/jpl/primitives_repo')
    exit()

# PRIMITIVES_REPO = '/home/ktyao/dev/dsbox/primitives_repo'

import json
import os

# from importlib import reload
# import dsbox.planner.common.library
# reload(dsbox.planner.common.library)
from dsbox.planner.common.library import D3MPrimitiveLibrary
 
# import dsbox.planner.common.ontology
# reload(dsbox.planner.common.ontology)
from dsbox.planner.common.ontology import D3MOntology

lib_dir = os.path.join(os.path.dirname(__file__), '../../../library2')
hierarchy_file = os.path.join(lib_dir, 'two_level_clustering.json')
profile_file = os.path.join(lib_dir, 'profile_output.json') 

# Create library with all D3M primitives
d3m_primtive_library = D3MPrimitiveLibrary()
d3m_primtive_library.load_from_directory(PRIMITIVES_REPO)

# Define hierarchy over all D3M primitives
onto = D3MOntology(d3m_primtive_library)
onto.load_curated_hierarchy(hierarchy_file)

# Augment primitive metadata with Daniel's primitive profiler output 
with open(profile_file) as fp:
    primitive_profiles = json.load(fp)

all_preconditions = set()
for package, profile in primitive_profiles.items():
    if not d3m_primtive_library.has_primitive_by_package(package):
        print('Cannot find class: {}'.format(package))
        continue
    primitive = d3m_primtive_library.get_primitive_by_package(package)
    if 'Requirements' in profile:
        all_preconditions |= {x for x in profile['Requirements']}
        primitive.addPrecondition({x : True 
                                   for x in profile['Requirements']})
    if 'Error' in profile:
        primitive.addErrorCondition({x:True for x in profile['Error']})

            



