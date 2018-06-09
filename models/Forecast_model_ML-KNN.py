import pandas as pd
import numpy as np
import datetime
import time
import os

import subprocess
from io import StringIO   # StringIO behaves like a file object

from sklearn import svm, ensemble, neighbors, tree
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold


###############################################################################
# Scikit Learn optimization
def ScikitLearn(X_train, X_test, y_train, y_test, params):

	# Scale data (mean=0 / std=1)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	# Create an instance of Regressor and fit the data.
	clf = neighbors.KNeighborsRegressor(**params)
	clf.fit(X_train, y_train)

	# Make prediction
	y_pred = clf.predict(X_test)
	numSamples = len(y_test)

	# MAE (Mean Absolute Error)
	mae = np.sum(np.fabs(y_test - y_pred)) / numSamples
	return mae


#######################################################################
def Statistics(boolVariables, listFiles, kf, params):

	list_MAE = np.empty(0)

	for train_index, test_index in kf.split(data):

		# Define target/feature
		X_train, X_test = data[train_index,1:], data[test_index,1:]
		y_train, y_test = data[train_index,0], data[test_index,0]

		# Scikit-Learn model
		mae = ScikitLearn(X_train, X_test, y_train, y_test, params)
		list_MAE = np.append(list_MAE, mae)

	return np.mean(list_MAE)


#######################################################################
if __name__ == "__main__":

	ml_Folder = './'
	logFolder = './log'
	strFileAccuracy = logFolder + '/ECMWF_ML-KNN.txt'

	# Log folder
	if not os.path.exists(logFolder):
		os.makedirs(logFolder)

	# k-Folds
	numFolds = 3
	kf = KFold(n_splits=numFolds, shuffle=False, random_state=0)

	data = ...

	# Iterate over hyper-parameters
	arrayNeighbors = np.array([1,2,5,10,20,50,100])
	arrayWeights = np.array(['uniform', 'distance'])

	for idNeighbors, numNeighbors in enumerate(arrayNeighbors):
		for idWeights, strWeights in enumerate(arrayWeights):

			# Statistical values
			print 'n_neighbors: ', numNeighbors, ' - weights: ', strWeights
			params = {'n_neighbors': numNeighbors, 'weights': strWeights}
			mae = Statistics(kf, data, params)

			# Store the rejected variable and the MAE obtained
			file = open(strFileAccuracy, 'a')
			s = '%.3f\t%s\n' % (mae, params)
			print '\n============================================================'
			print 'Store: ', s
			print '============================================================\n'
			file.write(s)
			file.close()

	####################

