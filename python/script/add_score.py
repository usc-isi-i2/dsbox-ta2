import argparse
import json
import pprint
import sys

import numpy
import pandas as pd

from pathlib import Path

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

name_map = {
    'ACCURACY': 'accuracy',
    'PRECISION': 'precision',
    'RECALL': 'recall',
    'F1': 'f1',
    'F1_MICRO': 'f1Micro',
    'F1_MACRO': 'f1Macro',
    'ROC_AUC': 'rocAuc',
    'ROC_AUC_MICRO': 'rocAucMicro',
    'ROC_AUC_MACRO': 'rocAucMacro',
    'MEAN_SQUARED_ERROR': 'meanSquaredError',
    'ROOT_MEAN_SQUARED_ERROR': 'rootMeanSquaredError',
    'MEAN_ABSOLUTE_ERROR': 'meanAbsoluteError',
    'R_SQUARED': 'rSquared',
    'NORMALIZED_MUTUAL_INFORMATION': 'normalizedMutualInformation',
    'JACCARD_SIMILARITY_SCORE': 'jaccardSimilarityScore',
    'PRECISION_AT_TOP_K': 'precisionAtTopK',
    'OBJECT_DETECTION_AVERAGE_PRECISION': 'objectDetectionAP',
    'HAMMING_LOSS': 'hammingLoss'
}

def to_dict(row):
    result = dict(row)
    result['metric'] = name_map[result['metric']]
    return result

def add_score(score_dir, info_dir):
    for score_csv in score_dir.glob('*.csv'):
        score_df = pd.read_csv(score_csv)
        score = [to_dict(score_df.iloc[i, :]) for i in range(score_df.shape[0])]
        info_file = info_dir / Path(score_csv.stem).with_suffix('.json')
        if info_file.exists():
            with open(info_file, 'r') as fp:
                info = json.load(fp)
            info['score'] = score
            with open(info_file, 'w') as fp:
                json.dump(info, fp, indent=4, cls=MyEncoder)
        else:
            print('Missing info file:', info_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add score to pipeline info",
    )
    parser.add_argument('score_dir', help="Directory containing *.score.csv files")
    parser.add_argument('info_dir', help="Directory containing *.json info files")

    args = parser.parse_args()
    score_dir = Path(args.score_dir)
    info_dir = Path(args.info_dir)

    if not score_dir.exists():
        print('Missing score dir:', score_dir)
        sys.exit(1)

    if not info_dir.exists():
        print('Missing score dir:', info_dir)
        sys.exit(1)

    add_score(score_dir, info_dir)
