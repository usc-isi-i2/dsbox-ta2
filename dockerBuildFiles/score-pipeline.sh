#!/bin/bash

shopt -s nullglob

if [ ! -d /output/pipelines_scored ]; then
    echo Missing /output/pipelines_scored
    exit
fi
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
