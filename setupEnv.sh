#!/usr/bin/env bash

# Update d3m module to "master" branch version
cd ../d3m
git checkout master
git pull
pip install -e .
cd -

# Update common-primitives to "master" branch version
cd ../common-primitives
git checkout master
git pull
pip install -e .
cd -

# Update sklearn-wrap to "dist" branch version
cd ../sklearn-wrap
git checkout dist
git pull
pip install -e .
cd -

# Update dsbox-ta2 to "template-2018-june" branch version

# I kind of assume you do it first to get this file anyway

#########################################################

# 1. Replace the "base.py" and "pipeline.py" part at d3m/metadata
cp d3m/base.py ../d3m/metadata
cp d3m/pipeline.py ../d3m/metadata

# 2. Replace the " entry_points.ini" part at common_primitives
cp common_primitives/entry_points.ini ../common-primitives


# 3. Replace the "utils.py" and "denormalize.py" part at
# common_primitives/common_primitives
cp common_primitives/{denormalize.py,utils.py} ../common-primitives/common_primitives

# 4. After replacing these things, go back to common-primitives folder
# and run "pip install -e ." to reinstall the common-primitive module.

cd ../common-primitives
pip install -e .
cd -

# 5. Then, you can run "$ python -m d3m.index search" to check whether
# the "denormalize" primitive appeared in the search results.
echo "CHECK THIS OUTPUT"
python -m "d3m.index.search"
echo "-------------------"