import os
import re

class EnsInfo:
	def __init__(self, dataset_name, raw_improve = float('nan'), pct_improve = float('nan'), metric_name = '', blank = False, pipeline_error = False):
		self.dataset_name = dataset_name
		self.raw_improve = raw_improve
		self.pct_improve = pct_improve
		self.metric_name = metric_name
		self.blank = blank
		self.pipeline_error = pipeline_error
    def to_csv_header(self, sep=','):
        return sep.join(['dataset_name', 'metric_name', 'raw_improve', 'pct_improve', 'blank', 'pipeline_error'])

    def to_csv(self, sep='\t'):
        return sep.join(['{}'.format(x)
                         for x in [self.dataset_name, self.metric_name, self.raw_improve,
                                   self.pct_improve, self.blank, self.pipeline_error]])
    def __str__(self):
        return 'EnsInfo({}, {}, {:.2f}, {:.2f}, {})'.format(
            self.dataset_name, self.metric_name, self.raw_improve,
                                   self.pct_improve, self.blank)

def get_ensembles(outputs_dir = './outputs'):
	for datadir in os.listdir(outputs_dir):
		pipeline_file = os.path.join(outputs_dir, datadir, 'temp', 'ensemble.txt')
		ensemble_file = os.path.join(outputs_dir, datadir, 'temp', 'ensemble.txt')
		if os.path.exists(ensemble_file):
			with open(pipeline_file) as fp:
				line_count = sum(1 for _ in f)
				if line_count == 0 or line_count == 1:
					pipeline_error = True
				else:
					pipeline_error = False

			with open(ensemble_file) as f:
				line_count = sum(1 for _ in f)
				if line_count == 0:
					datasets.append(EnsInfo(datadir, blank = True))
				else: 
					for line in f:
						stats = re.split(' : |, | , |, | : ', line)
						datasets.append(EnsInfo(datadir, raw_improve = float(stats[1].strip()), pct_improve = float(stats[2].strip()), metric_name = str(stats[-1])))
	with open('ensemble_results.csv', 'w') as f:
		f.write(datasets[0].to_csv_header())
		f.write('\n')
		for info in datasets:
			f.write(info.to_csv())
			f.write('\n')

if __name__ == '__main__':
	get_ensembles()

