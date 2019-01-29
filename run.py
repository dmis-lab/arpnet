from __future__ import unicode_literals, print_function, division
import numpy as np
import tensorflow as tf
import time
import math
import sys
import parameters
import utils
import models
import argparse

def load_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=int, default=1, help='Choice the task')
	args = parser.parse_args()

	return args

# =============================================
def print_hyper_parameter():
	print("======================================================")
	print("HYPER-PARAMETERS\n")
	print("HIDDEN_LAYER: {}".format(parameters.hidden_layer))
	print("DEMO_FEATURE: {}".format(parameters.demo_feature))
	print("BIO_FEATURE: {}".format(parameters.bio_feature))
	print("GENE_FEATURE: {}".format(parameters.gene_feature))
	print("METHYL_FEATURE: {}".format(parameters.methyl_feature))
	print("BATCH_SIZE: {}".format(parameters.batch_size))
	print("LEARNING_RATE: {}".format(parameters.learning_rate))
	print("DROPOUT_RATE: {}".format(parameters.dropout_rate))


def dataprocessing(task):
	# =============================================
	# DATA PREPARATION
	# =============================================
	print("======================================================")
	print("Data Loading..")
	patient_data, train, test = utils.load_data(
		parameters.patient_data_path, parameters.train_data_path,
		parameters.test_data_path, task)

	patient_header, patient_data = patient_data[0], patient_data[1:]

	print("Make Dictionary..")
	drug2id, id2drug, patient_data = utils.make_dict(train, test, patient_data)

	print("Make Input..")
	train, test = utils.make_input(train, test, patient_data, drug2id, task)

	print("======================================================")
	print("DATA STATISTICS\n")
	print("NUMBER OF PATIENTS: {}".format(len(patient_data)))
	print("TRAIN DATA: {}".format(len(train)))
	print("TEST DATA: {}".format(len(test)))
	print("NUMBER OF DRUGS: {}".format(len(drug2id)))

	return patient_data, train, test, drug2id

print("======================================================")
print("Tensorflow Log..")

# def task_1(train, test, model):

def train_step(sess, model, train):
	if len(train) % parameters.batch_size == 0:
		batch_num = int(len(train)/parameters.batch_size)
	else:
		batch_num = int(len(train)/parameters.batch_size) + 1

	step = 0
	loss = .0

	batches = utils.batches(train, parameters.batch_size)
	for batch in batches:
		input_x, hamds, intervals, drugs, targets = batch
		_loss, _ = sess.run([model.loss, model.optimizer],
						feed_dict={model.input_x: input_x, 
					   			   model.input_hamd: hamds, 
					   			   model.input_interval: intervals, 
					   			   model.input_drug: drugs, 
					   			   model.input_target: targets,
					   			   model.dropout_rate: parameters.dropout_rate})
		loss += _loss
		step += 1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Training Batch: [{}/{}]".format(step, batch_num))

	return loss

def task_1(sess, model, test):
	step = 0
	if len(test) % parameters.batch_size == 0:
		batch_num = int(len(test)/parameters.batch_size)
	else:
		batch_num = int(len(test)/parameters.batch_size) + 1

	targets = []
	inferences = []

	batches = utils.batches(test, parameters.batch_size)
	for batch in batches:
		input_x, hamds, intervals, drugs, _targets = batch
		inference = sess.run(model.inference, 
					feed_dict={model.input_x: input_x, 
				   			   model.input_hamd: hamds, 
				   			   model.input_interval: intervals, 
				   			   model.input_drug: drugs,
				   			   model.dropout_rate: 1.0})
		targets.append(_targets)
		inferences.append(inference)

	targets = np.concatenate(targets, 0)
	inferences = np.concatenate(inferences, 0)

	rms = math.sqrt(sum((targets - inferences)**2)/len(targets))

	return rms

def task_2(sess, model, test):
	step = 0
	tp, tn, cp, cn, pcp = .0, .0, .0, .0, .0

	if len(test) % parameters.batch_size == 0:
		batch_num = int(len(test)/parameters.batch_size)
	else:
		batch_num = int(len(test)/parameters.batch_size) + 1

	batches = utils.batches_eval(test)
	for batch in batches:
		input_x, initial, drugs, intervals, label, = batch

		hamd = initial
		for m in xrange(len(drugs)):
			hamd = sess.run(model.inference, feed_dict={
			model.input_x: input_x,
			model.input_hamd: hamd, 
			model.input_interval: [intervals[m]], 
			model.input_drug: [drugs[m]],
			model.dropout_rate: 1.0})

		if (float(initial[0]) - float(hamd[0]))/float(initial[0]) \
		>= parameters.threshold:
			_label = True
		else:
			_label = False

		if _label == True:
			pcp += 1
			if label == True:
				tp += 1

		if _label == False:
			if label == False:
				tn += 1

		if label == True:
			cp += 1
		else:
			cn += 1

		step += 1
		
	sensitivity = tp/cp
	specificity = tn/cn
	acc = (tp+tn)/(cp+cn)
	if pcp != 0:
		precision = tp/pcp
	else:
		precision = .0

	if sensitivity + precision != 0:
		f1 = 2*sensitivity*precision/(sensitivity + precision)
	else:
		f1 = .0

	return sensitivity, specificity, precision, f1, acc


def main(task):
	print_hyper_parameter()
	patient_data, train, test, drug2id = dataprocessing(task)

	with tf.Graph().as_default():
	    session_conf = tf.ConfigProto(allow_soft_placement=True, 
	    							  log_device_placement=False, 
	    							  gpu_options=tf.GPUOptions(allow_growth=True))
	    sess = tf.Session(config=session_conf)

	    with sess.as_default():
			print("======================================================")
			print("Initializing..")

			model = models.MODEL(parameters.hidden_layer, 
								 parameters.embedding_dim,
								 len(drug2id), 
								 parameters.demo_feature, 
								 parameters.bio_feature, 
								 parameters.gene_feature, 
								 parameters.methyl_feature, 
								 parameters.learning_rate, 
								 parameters.dropout_rate)

			# ==============================
			# Variables Initializer
			# ==============================				
			initializer = tf.global_variables_initializer()
			sess.run(initializer)

			print("======================================================")
			print("Training..\n")

			count = 0
			for i in xrange(parameters.epochs):
				loss = train_step(sess, model, train)

				sys.stdout.write("\033[F")
				sys.stdout.write("\033[K")
				print("Process Training Epoch: [{}/{}] Loss: {}\n"\
					.format(i+1, parameters.epochs, loss))

				if (i+1) % 5 == 0:
					sys.stdout.write("\033[F")
					sys.stdout.write("\033[K")
					print("Evaluation..")

					if task == 1:
						rms = task_1(sess, model, test)
						
						sys.stdout.write("\033[F")
						sys.stdout.write("\033[K")
						print("Process Evaluation Epoch: [{}/{}] RMS: {}\n"\
							.format(i+1, parameters.epochs, rms))

					else:
						scores = task_2(sess, model, test)

						sys.stdout.write("\033[F")
						sys.stdout.write("\033[K")
						print("Process Evaluation Epoch: [{}/{}] (Sen/Spe/Pre/F1/Acc): {}\n"\
							.format(i+1, parameters.epochs, scores))





# total_rms, total_r2 = .0, .0
# for j in xrange(5):
# 	rms, r2 = [], []
# 	for i in xrange(1, 11):
# 		print("{}-fold".format(i))
# 		train, validation, drug2id = dataprocessing(i)
# 		_rms, _r2 = main()
# 		rms.append(_rms)
# 		r2.append(_r2)

# 	total_rms += sum(rms)/len(rms)
# 	total_r2 += sum(r2)/len(r2)

# print(total_rms/5, total_r2/5)

if __name__ == "__main__":
	args = load_argument()
	main(args.task)
