import os
import pandas as pd

all_data = "/nfs1/dsbox-repo/data/datasets/training_datasets/LL0/"

targets = dict()

for dataset in os.listdir(all_data):
	curr_dataset = all_data + dataset

	try:
		score_file = ""
		for folder in os.listdir(curr_dataset):
			if 'solution' in folder:
				score_file = curr_dataset + "/" + folder + "/scores.csv"

		df = pd.DataFrame.from_csv(score_file)
		target_score = df['value'][0]
	except:
		print("failed for", dataset)

	targets[dataset] = target_score


our_scores = dict()
for fn in os.listdir("./output/"):
	# avoid folders, only read files
	if ".txt" not in fn:
		continue

	try:
		f = open("./output/" + fn, "r")
		score = float(f.readlines()[2])
		# print(score)
		f.close()

		our_scores[fn.rsplit(".")[0]] = score
	except:
		print("failed for", fn)

# generate tuples from the data we have processed
statistics = []
for ds in targets:
	if ds in our_scores:
		statistics.append((ds, our_scores[ds], targets[ds]))
	else:
		statistics.append((ds, "N/A", targets[ds]))


df = pd.DataFrame(statistics, columns=['dataset', 'our_score', 'target'])
print(df.head())

df.to_csv("system_results.csv")





	