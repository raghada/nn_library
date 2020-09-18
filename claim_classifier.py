import os
import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten)
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight, resample
import matplotlib.pyplot as plt
import itertools
#from keras.models import load_model


class ClaimClassifier:

	def __init__(self, data_raw, learningRate, n_layers, n_units):

		self.model = Sequential()

		self.model.add(Dense(n_units[0], input_shape=(9,), activation="relu"))
		self.model.add(BatchNormalization())
		for i in range(1, len(n_units)):
			self.model.add(Dense(n_units[i], activation="relu"))
			self.model.add(BatchNormalization())

		self.model.add(Dense(1, activation='sigmoid'))

		self.model.compile(optimizer=Adam(lr=learningRate),
			loss='binary_crossentropy', metrics=['binary_accuracy'])

		#self.model.summary()
		
		self.scaler = StandardScaler().fit(data_raw)


	def _preprocessor(self, X_raw):
		"""Data preprocessing function.

		This function prepares the features of the data for training,
		evaluation, and prediction.

		Parameters
		----------
		X_raw : numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
			A numpy array, this is the raw data as downloaded

		Returns
		-------
		X: numpy.ndarray (NOTE, IF WE CAN USE PANDAS HERE IT WOULD BE GREAT)
			A clean data set that is used for training and prediction.
		"""
		return self.scaler.transform(X_raw)


	def fit(self, X_raw, y_raw, X_val, y_val, epochs, batchSize=32):
		"""Classifier training function.

		Here you will implement the training function for your classifier.

		Parameters
		----------
		X_raw : numpy.ndarray
			A numpy array, this is the raw data as downloaded
		y_raw : numpy.ndarray (optional)
			A one dimensional numpy array, this is the binary target variable

		Returns
		-------
		?
		"""

		# YOUR CODE HERE
		X = pd.concat([X_val,y_val],axis=1)

		class_0 = X[X['made_claim']==0]
		class_1 = X[X['made_claim']==1]
		class_1_over = resample(class_1,replace=True,n_samples=len(class_0),random_state=1)

		df_val = pd.concat([class_0,class_1_over])

		X_val = df_val.iloc[:,:9]
		y_val = df_val['made_claim']

		X_train_clean = self._preprocessor(X_raw)
		X_val_clean = self._preprocessor(X_val)


		hist = self.model.fit(X_train_clean, y_raw, epochs=epochs, batch_size=batchSize,
			validation_data=(X_val_clean,y_val),
			shuffle=True,verbose=False,
			callbacks=[ModelCheckpoint('./cur_model_part2.h5',save_best_only=True),
			ReduceLROnPlateau()])

		self.model = keras.models.load_model('./cur_model_part2.h5')
		os.remove('cur_model_part2.h5')

		return hist


	def predict(self, X_raw):
		"""Classifier probability prediction function.

		Here you will implement the predict function for your classifier.

		Parameters
		----------
		X_raw : numpy.ndarray
			A numpy array, this is the raw data as downloaded

		Returns
		-------
		numpy.ndarray
			A one dimensional array of the same length as the input with
			values corresponding to the probability of beloning to the
			POSITIVE class (that had accidents)
		"""

		x_clean = self._preprocessor(X_raw)

		output = self.model.predict_classes(x_clean)
		return output

	def evaluate_architecture(self, X_raw, y_raw):
		"""Architecture evaluation utility.

		Populate this function with evaluation utilities for your
		neural network.

		You can use external libraries such as scikit-learn for this
		if necessary.
		"""
		prediction = self.predict(X_raw)

		cfm = confusion_matrix(y_raw,prediction)

		metrics = precision_recall_fscore_support(y_raw,prediction)

		f1 = f1_score(y_raw,prediction)

		return cfm, metrics, f1


	def save_model(self):
		with open("part2_claim_classifier.pickle", "wb") as target:
			pickle.dump(self, target)



def ClaimClassifierHyperParameterSearch(X_train,y_train,X_test,y_test,X_val,y_val):
	"""Performs a hyper-parameter for fine-tuning the classifier.

	Implement a function that performs a hyper-parameter search for your
	architecture as implemented in the ClaimClassifier class.

	The function should return your optimised hyper-parameters.
	"""

	best_f1 = 0
	layer_widths = [8, 16, 32]
	n_units_possibilities = []
	for n_layers in [2, 3, 4]:
		for i in range(len(layer_widths)):
			comb = [s for s in itertools.combinations_with_replacement(layer_widths[:i+1], n_layers)]
			for n_units in comb:
				n_units_possibilities.append(n_units[::-1])

	for n_units in list(set(n_units_possibilities)):
		classifier = ClaimClassifier(X_train, learningRate=.001, n_layers=len(n_units), n_units=n_units)

		history = classifier.fit(X_train, y_train, X_val, y_val, epochs=100)

		cfm, metrics, f1 = classifier.evaluate_architecture(X_test,y_test)

		print('###############', n_layers, 'layers with', n_units, 'units')
		print(cfm)
		print('precision_recall_fscore_support')
		print(metrics)
		print('global f1-score')
		print(f1)
		print('###############')

		plt.clf()
		plt.plot(history.history['binary_accuracy'])
		plt.plot(history.history['val_binary_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		#plt.show()
		#plt.savefig('./results/accuracy_{}_layers_{}_units.png'.format(n_layers,n_units))

		plt.clf()
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		#plt.show()
		#plt.savefig('./results/loss_{}_layers_{}_units.png'.format(n_layers,n_units))

		if f1 > best_f1:
			best_f1 = f1
			best_hyper = (n_layers,n_units)
	return best_hyper


if __name__ == '__main__':
	# Load full dataset into pandas dataframe
	df = pd.read_csv('sample_claim_data.csv')

	# Split data
	X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:9], \
		df['made_claim'],test_size=.2,stratify=df['made_claim'],random_state=43558)

	X = pd.concat([X_train,y_train],axis=1)

	class_0 = X[X['made_claim']==0]
	class_1 = X[X['made_claim']==1]
	class_1_over = resample(class_1,replace=True,n_samples=len(class_0),random_state=43558)

	df_train = pd.concat([class_0,class_1_over])

	X_train = df_train.iloc[:,:9]
	y_train = df_train['made_claim']

	X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size=.5,stratify=y_test,random_state=43558)

	best_hyper = ClaimClassifierHyperParameterSearch(X_train,y_train,X_test,y_test,X_val,y_val)

	classifier = ClaimClassifier(X_train, learningRate=.0001, n_layers=best_hyper[0], n_units=best_hyper[1])

	history = classifier.fit(X_train, y_train, X_val, y_val, epochs=5)

	classifier.save_model()

	#print('Best hyper',best_hyper)
