#!/bin/bash

# This script:
# 1. Start TA2 in server mode
# 2. Runs dummy_ta3 client to connect to TA2 to generate pipelines
# 3. For each TA2 pipeline, train-test-score

shopt -s nullglob

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
echo "Score pipelines"
mkdir -p /output/score
cd /output/pipelines_ranked
if [ -d /input/SCORE/dataset_TEST ]; then
    score_doc=/input/SCORE/dataset_TEST/datasetDoc.json
elif [ -d /input/SCORE/dataset_SCORE ]; then
    score_doc=/input/SCORE/dataset_SCORE/datasetDoc.json
fi
for f in *.json; do
    output_file_name=${f%.*}
    echo "python3 -m d3m runtime -v /static fit-score -p $f -r /input/TRAIN/problem_TRAIN/problemDoc.json -i /input/TRAIN/dataset_TRAIN/datasetDoc.json  -t /input/TEST/dataset_TEST/datasetDoc.json -a $score_doc -o /output/predictions/${output_file_name}.csv -c /output/score/${output_file_name}.score.csv &> /output/logs/scoring-${output_file_name}.out"
done | xargs -P5 -l1 -I {} bash -c "{}"

for f in *.json; do
    output_file_name=${f%.*}
    if [[ ! -f /output/predictions/${output_file_name}.csv || ! -s /output/predictions/${output_file_name}.csv ]]; then
        echo Failed to generate prediction for $f
        continue
    fi
    if [[ ! -f /output/score/${output_file_name}.score.csv || ! -s /output/score/${output_file_name}.score.csv ]]; then
        echo Failed to generate score for $f
        continue
    fi
done

# Add score to pipeline info
python3 /user_opt/dsbox/dsbox-ta2/python/script/add_score.py /output/score /output/pipelines_info

echo "Cleaning up"
/bin/rm -R /output/pipelines_fitted/*/*.pkl

echo "Exiting"
kill $ta2_pid
kill -9 $ta2_pid
