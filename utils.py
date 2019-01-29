import csv
import random
import numpy as np

def load_data(patient_data_path, train_data_path, test_data_path, task):
	patient_data = []
	with open(patient_data_path) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			patient_data.append(line)

	train = []
	with open("{}_{}.tsv".format(train_data_path, task)) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			train.append(line)

	test = []
	with open("{}_{}.tsv".format(test_data_path, task)) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		for line in tsvreader:
			test.append(line)


	return patient_data, train, test

def make_dict(train, validation, _patient_data):
	drug2id = {"":0}
	id2drug = [""]

	for line in train+validation:
		drugs = line[-2].split(";")
		for drug in drugs:
			if drug not in drug2id and drug != "":
				drug2id[drug] = len(drug2id)
				id2drug.append(drug)

	patient_data = {}
	for line in _patient_data:
		patient_data[line[0]] = line

	return drug2id, id2drug, patient_data

def make_input(_train, _test, patient_data, drug2id, task):
	train = []
	for line in _train:
		patient, hamd, interval, drugs, target = line
		drugs = [drug2id[drug] for drug in drugs.split(";")]

		train.append(([float(feature) for feature \
			in patient_data[patient][1:-2]], 
					   float(hamd),
					   float(interval),
					   drugs,
					   float(target)))

	if task == 1:
		test = []
		for line in _test:
			patient, hamd, interval, drugs, target = line
			drugs = [drug2id[drug] for drug in drugs.split(";")]

			test.append(([float(feature) for feature \
				in patient_data[patient][1:-2]], 
						   float(hamd),
						   float(interval),
						   drugs,
						   float(target)))

		return train, test

	else:
		test = {}
		for line in _test:
			if line[0] not in test:
				test[line[0]] = [line]
			else:
				test[line[0]].append(line)

		_test = {}
		for patient, value in test.items():
			
			initial = value[0][1]
			last = value[-1][-1]
			if (float(initial)-float(last))/float(initial) >= 0.5:
				label = True
			else:
				label = False
			
			drugs = []
			intervals = []
			for line in value:
				patient, hamd, interval, _drugs, target = line
				drugs.append([drug2id[drug] for drug in _drugs.split(";")])
				intervals.append(float(interval))

			_test[patient] = {"input_x": [float(feature) for feature \
			in patient_data[patient][1:-2]],
							  "initial": float(initial),
							  "drugs": drugs,
							  "intervals": intervals,
							  "label": label}

		return train, _test


def batches(_data, batch_size):
	random.shuffle(_data)
	if len(_data) % batch_size == 0:
		batch_num = int(len(_data)/batch_size)
	else:
		batch_num = int(len(_data)/batch_size) + 1

	for i in xrange(batch_num):
		input_x = []
		hamds = []
		intervals = []
		drugs = []
		targets = []

		left = i*batch_size
		right = min((i+1)*batch_size, len(_data))

		for data in _data[left:right]:
			input_x.append(data[0])
			hamds.append(data[1])
			intervals.append(data[2])
			drugs.append(data[3])
			targets.append(data[4])

		yield input_x, hamds, intervals, drugs, targets

def batches_eval(_data):
	for _, value in _data.items():
		yield [value["input_x"]], [value["initial"]], value["drugs"], \
		value["intervals"], value["label"]


def get_scores(predicted, targets):
	score = np.square(predicted - targets)
	score = np.mean(score)
	score = np.sqrt(score)

	return score