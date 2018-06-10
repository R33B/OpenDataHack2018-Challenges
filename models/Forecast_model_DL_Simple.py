# -*- coding: utf-8 -*-
import os
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

##############################################################################
def Clean_DST_errors(df):
	df.loc[df['wind'] == 'False', 'wind'] = 'nan'
	df['wind'] = df['wind'].astype(np.float)
	df.dropna(inplace=True)
	df['datetime'] = pd.to_datetime(df['datetime'])
	return df

#######################################################################
# Kernel: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def getModel_Sim(numRow, numCol):
	# Building the model
	gmodel=Sequential()
	# Conv Layer 1
	gmodel.add(Conv2D(16, kernel_size=(3, 3),activation='relu', input_shape=(numRow, numCol, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(0.0))
	# Conv Layer 2
	gmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.0))
	# Conv Layer 3
	gmodel.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.0))
	# Flatten the data for upcoming dense layers
	gmodel.add(Flatten())
	# Dense Layers
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.0))
	# Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('linear'))
	# Compile
	mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='mean_absolute_error', optimizer=mypotim, metrics=['mean_absolute_error'])
#	gmodel.summary()
	return gmodel


def getModel_Adv(numRow, numCol):
	# Building the model
	gmodel=Sequential()
	# Conv Layer 1
	gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(numRow, numCol, 3)))
	gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	# Conv Layer 2
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	# Conv Layer 3
	gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	# Conv Layer 4
	gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	gmodel.add(Dropout(0.2))
	# Flatten the data for upcoming dense layers
	gmodel.add(Flatten())
	# Dense Layers
	gmodel.add(Dense(512))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))
	# Dense Layer 2
	gmodel.add(Dense(256))
	gmodel.add(Activation('relu'))
	gmodel.add(Dropout(0.2))
	# Sigmoid Layer
	gmodel.add(Dense(1))
	gmodel.add(Activation('linear'))
	# Compile
	mypotim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	gmodel.compile(loss='mean_absolute_error', optimizer=mypotim, metrics=['mean_absolute_error'])
#	gmodel.summary()
	return gmodel

def get_callbacks(filepath, patience=2):
	es = EarlyStopping('val_loss', patience=patience, mode="min")
	msave = ModelCheckpoint(filepath, save_best_only=True)
	return [es, msave]

file_path = "./models/Forecast_model_DL_Simple_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

#######################################################################
def train_Model():

	# Read real Wind energy generation
	list_Files_Generation = glob.glob('./data/TSO_ESP_Generation_*.tsv')
	df_Generation = pd.concat([pd.read_csv(strFile, sep='\t') for strFile in list_Files_Generation], ignore_index=True)
	df_Generation = Clean_DST_errors(df_Generation)
	df_Generation = df_Generation.drop(['timestamp'], axis=1)
	df_Generation.set_index('datetime', inplace=True)

	# Read the json files into a pandas dataframe
	dateTrain_fin = datetime.datetime(2017,6,1)
	list_Files_CDS = glob.glob('./data/json/*.json')
	list_Files_CDS = [strFile for strFile in list_Files_CDS if (datetime.datetime.strptime(strFile.split('/')[-1], 'dataset_original_%Y-%m.json') < dateTrain_fin)]
	df = pd.concat([pd.read_json(strFile) for strFile in list_Files_CDS], ignore_index=True)
	df['datetime'] = pd.to_datetime(df['datetime'])
	df.set_index('datetime', inplace=True)
	df['target'] = df_Generation['wind']
	df.dropna(inplace=True)

	numRow = df.iloc[0]['numRow']
	numCol = df.iloc[0]['numCol']

	# Initialize model
	gmodel = getModel_Sim(numRow, numCol)

	# Load HDF5 weights (if exists)
	strFile_HDF5 = './models/Forecast_model_DL_Simple_weights.hdf5'
	if os.path.isfile(strFile_HDF5):
		gmodel.load_weights(strFile_HDF5)

	# Cross validation training
	u_100 = np.array([np.array(var).astype(np.float32).reshape(numRow, numCol) for var in df['u_100']])
	v_100 = np.array([np.array(var).astype(np.float32).reshape(numRow, numCol) for var in df['v_100']])
	ws_100 = np.sqrt(u_100**2 + v_100**2)
	X = np.concatenate([u_100[:, :, :, np.newaxis],
								v_100[:, :, :, np.newaxis],
								ws_100[:, :, :, np.newaxis]], axis=-1)
	y = df['target']

	# Deep Learning
	X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X, y, random_state=1, train_size=0.75)
	gmodel.fit(X_train_cv, y_train_cv,
       batch_size=24,
       epochs=50,
       verbose=1,
       validation_data=(X_valid, y_valid),
       callbacks=callbacks)


#######################################################################
def predict_Model(dateIni, dateFin, modelType):

	# Read real Wind energy generation
	list_Files_Generation = glob.glob('./data/TSO_ESP_Generation_*.tsv')
	df_Generation = pd.concat([pd.read_csv(strFile, sep='\t') for strFile in list_Files_Generation], ignore_index=True)
	df_Generation = Clean_DST_errors(df_Generation)
	df_Generation = df_Generation.drop(['timestamp'], axis=1)
	df_Generation.set_index('datetime', inplace=True)

	# Read the json files into a pandas dataframe
	list_Files_CDS = glob.glob('./data/json/*.json')
	df = pd.concat([pd.read_json(strFile) for strFile in list_Files_CDS], ignore_index=True)
	df['datetime'] = pd.to_datetime(df['datetime'])
	df.set_index('datetime', inplace=True)
	df['target'] = df_Generation['wind']

	numRow = df.iloc[0]['numRow']
	numCol = df.iloc[0]['numCol']

	# Initialize model
	if(modelType == 'simple'):
		gmodel = getModel_Sim(numRow, numCol)
		gmodel.load_weights('./models/Forecast_model_DL_Simple_weights_Sim.hdf5')
	else:
		gmodel = getModel_Adv(numRow, numCol)
		gmodel.load_weights('./models/Forecast_model_DL_Simple_weights_Adv.hdf5')

	# Predict
	df_test = df.loc[(df.index.year == 2017) & (df.index.month >= 6)]
	u_100_test = np.array([np.array(var).astype(np.float32).reshape(numRow, numCol) for var in df_test['u_100']])
	v_100_test = np.array([np.array(var).astype(np.float32).reshape(numRow, numCol) for var in df_test['v_100']])
	ws_100_test = np.sqrt(u_100_test**2 + v_100_test**2)
	X_test = np.concatenate([u_100_test[:, :, :, np.newaxis],
										v_100_test[:, :, :, np.newaxis],
										ws_100_test[:, :, :, np.newaxis]], axis=-1)

	y_pred = gmodel.predict(X_test).squeeze()
	df_test['pred'] = y_pred


	# Time series of predicted values for 2018
	df = pd.DataFrame(columns=['real', 'pred', 'tmhrzn'])

	dateRef = dateIni
	while(dateRef < dateFin):

		print '===== dateRef: ', dateRef, ' ====='

		# Real wind energy generation for following 48 hours
		dateReal_ini = dateRef
		dateReal_fin = dateRef + datetime.timedelta(days=2)
		data_Real = df_Generation.loc[(df_Generation.index >= dateReal_ini) & (df_Generation.index < dateReal_fin), 'wind'].values

		# Forecast (Today and Tomorrow, the real values time-serie from yesterday)
		datePred_ini = dateRef
		datePred_fin = dateRef + datetime.timedelta(days=2)
		data_Pred = df_test.loc[(df_test.index >= datePred_ini) & (df_test.index < datePred_fin), 'pred'].values

		# Append
		if( len(data_Real) == len(data_Pred) ):
			df = df.append(pd.DataFrame({'real': data_Real, 'pred': data_Pred, 'tmhrzn': range(48)}), ignore_index=True)

		# Next time step
		dateRef = dateRef + datetime.timedelta(days=1)

	return df


#######################################################################
if __name__ == "__main__":

	# Train the model parameters
	train_Model()

#	dateIni = datetime.datetime(2017,6,1)
#	dateFin = datetime.datetime(2017,12,31)
#	df_Keras = predict_Model(dateIni, dateFin, modelType='simple')

