from __future__ import unicode_literals, print_function, division
import numpy as np
import tensorflow as tf

class MODEL(object):
	def __init__(self, hidden_layer, embedding_dim, drug_num, 
		demo_feature_num, bio_feature_num, gene_feature, 
		methyl_feature_num, learning_rate, dropout):
		
		# ==============================
		# Hyper-Parameters
		# ==============================
		self.hidden_layer = hidden_layer
		self.embedding_dim = embedding_dim
		self.input_num = demo_feature_num+bio_feature_num+methyl_feature_num\
		+gene_feature
		self.learning_rate = learning_rate
		# ==============================
		# Hyper-Parameters
		# ==============================
		self.drug_num = drug_num

		# ==============================
		# Embedding
		# ==============================
		self.zero_padding = tf.constant(0.0, 
										shape=[1, self.embedding_dim], 
										dtype=tf.float32)
		self.drug_embedding = tf.Variable(
			tf.truncated_normal([self.drug_num-1, self.embedding_dim]), 
			trainable=True, name="drug_embedding")
		self.drug_embedding = tf.concat(
			[self.zero_padding, self.drug_embedding], 0)

		# ==============================
		# Placeholders
		# ==============================
		self.input_x = tf.placeholder(tf.float32, 
									  [None, self.input_num], 
										 name="input_x")
		self.input_hamd = tf.placeholder(tf.float32, 
										 [None], 
										 name="input_hamd")
		self.input_interval = tf.placeholder(tf.float32, 
											 [None], 
											 name="input_interval")
		self.input_drug = tf.placeholder(tf.int32, 
										 [None, 2], 
										 name="input_drug")
		self.input_target = tf.placeholder(tf.float32, 
										   [None], 
										   name="input_target")	

		self.dropout_rate = tf.placeholder(tf.float32, 
										   name="input_target")	
		# ==============================
		# Parameters
		# ==============================
		self.hidden_w = tf.get_variable("hidden_w", 
										shape=[self.input_num+2,
											   self.hidden_layer], 
										initializer\
										=tf.contrib.layers.xavier_initializer())

		self.hidden_b = tf.Variable(tf.constant(0.1, 
												shape=[self.hidden_layer]), 
									name="hidden_b")
		
		self.output_w = tf.get_variable("output_w", 
										shape=[self.hidden_layer \
										+ self.embedding_dim, 1], 
										initializer\
										=tf.contrib.layers.xavier_initializer())

		self.output_b = tf.Variable(tf.constant(0.0, 
												shape=[1]), 
									name="output_b")

		# ==============================
		# Graph
		# ==============================
		self.user_representation = self._user_embedding()
		self.antidepressant_prescription_representation \
		= self._antidepressant_prescription_embedding() 
		self.inference = self._forward()
		self.loss = self._get_loss()
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)\
		.minimize(self.loss)

	def _user_embedding(self):
		user_representation = tf.concat([self.input_x,
										 tf.expand_dims(self.input_hamd,1), 
										 tf.expand_dims(self.input_interval,1)],
										 1)
		hidden_layer = tf.nn.xw_plus_b(user_representation, 
									   self.hidden_w, self.hidden_b)
		hidden_layer = tf.nn.relu(hidden_layer)
		
		return hidden_layer

	def _antidepressant_prescription_embedding(self):
		antidepressant_embedding = tf.nn.embedding_lookup(self.drug_embedding, 
														  self.input_drug)
		antidepressant_embedding = tf.reduce_sum(antidepressant_embedding, 1)
		
		return antidepressant_embedding

	def _forward(self):
		input_x = tf.concat([self.user_representation, 
							  self.antidepressant_prescription_representation]
							  , 1)
		input_x = tf.nn.dropout(input_x, self.dropout_rate)

		output = tf.nn.xw_plus_b(input_x, self.output_w, self.output_b)
		output = tf.reshape(output, [-1])
		
		return output

	def _get_loss(self):
		mse = tf.reduce_sum(tf.square(self.input_target - self.inference))
		return mse