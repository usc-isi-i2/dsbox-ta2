#!/bin/bash

# This script:
# 1. Start TA2 in server mode
# 2. Runs dummy_ta3 client to connect to TA2
# 3. Generate prediction for each TA2 pipeline
# 4. Generate score for each prediction

# 1. Start TA2 in server mode
/user_opt/d3mStart.sh &> /output/out.txt &
ta2_pid=$!

echo TA2 PID is $ta2_pid

# Wait for TA2 server to start
sleep 30

mkdir -p /output/predictions
mkdir -p /output/pipeline_runs

# 2. Runs dummy_ta3 client to connect to TA2
python3 -m dummy_ta3.dummy_ta3 -p /input/TRAIN/problem_TRAIN/problemDoc.json -d /input -e 127.0.0.1 -t $D3MTIMEOUT

# 2.5 Stop TA2 system
kill $(ps aux | grep ta2-server.py | grep python | awk '{print $2}')

# 3. Generate prediction for each TA2 pipeline
echo "Generating predictions"
cd /output/pipelines_ranked
for f in *.json; do
    output_file_name=${f%.*}
    echo " -m d3m.runtime fit-produce -p $f -r /input/TRAIN/problem_TRAIN/problemDoc.json -i /input/TRAIN/dataset_TRAIN/datasetDoc.json  -t /input/TEST/dataset_TEST/datasetDoc.json  -o /output/predictions/${output_file_name}.csv"
done | xargs -l1 -P5 python3 &> /dev/null

# 4. Generate score for each prediction
echo "Generating scores"
# bash /user_opt/dsbox/src/nist_eval_output_validation_scoring/scripts/score_output.sh
bash /user_opt/dsbox/src/d3m-outputs/scripts/score_output.sh
chmod -R go+rwX /output/*

echo "Cleaning up"
/bin/rm -R /output/pipelines_fitted/*/*.pkl

echo "Exiting"
kill $ta2_pid
kill -9 $ta2_pid

exit 0
