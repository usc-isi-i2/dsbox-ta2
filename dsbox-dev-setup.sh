
# Setup python path for DSBox development environment

top_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH=$top_dir/dsbox-cleaning:$top_dir/dsbox-profiling:$top_dir/dsbox-ta2/python:$top_dir/dsbox-corex:$top_dir/ta3ta2-api:$top_dir/spider:$top_dir/primitive-interfaces:$top_dir/primitive-examples
echo $PYTHONPATH
