import json
import os

from pprint import pprint

if __name__ == "__main__":


	for fn in os.listdir("./output/"):
		if not ".txt" in fn:
			continue

		pipeline_folder = "./output/" + fn.rsplit(".")[0] + "/pipelines"

		for pipeline in os.listdir(pipeline_folder):
			with open(pipeline_folder + "/" + pipeline) as f:
				data = json.load(f)

			print(data['metric_value'])
			print(data['steps'][len(data['steps']) - 1]['primitive']['name'])
			# pprint(data)

		break