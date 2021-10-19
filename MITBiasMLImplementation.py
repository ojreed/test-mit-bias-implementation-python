"""
run code with $py -3.8 MITBiasMLImplementation.py


"""


import numpy as np
import tensorflow as tf
import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
#import warnings
import os
import argparse
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
from numpy.random import choice



class DataProcessor(object):
	def __init__(self, folder,mode,targetDS=None):
		"""
		Mode: 
			allows for different reference data to be pulled from pickel rather than processed
			0 - runs all processing from beggining 
			1 - skips to csv files being input
			2 - skips to process 
			3 - skips to trained models (WIP)
		"""
		self.mode = mode 
		self.targetDS = targetDS
		self.folder = folder
		self.dataArrayRaw = self.setup(self.folder,self.mode) 
		self.dataArray = self.process(self.dataArrayRaw,self.mode)
		if self.targetDS == None:
			self.targetDS = [x for x in range(len(self.dataArray))]
		self.dummyModelList = self.buildNetworks(self.dataArray,self.mode,self.targetDS)
		self.trainedModelList = self.trainNetworks(self.dummyModelList,self.dataArray,self.targetDS,self.mode)
		self.main()

	def main(self):
		dsPointer=0
		for x in range(len(self.dataArray)):
			if x in self.targetDS:
				print(self.trainedModelList[dsPointer].predict(np.array(self.dataArray[x][0]).reshape(len(self.dataArray[x][0]),1,1)))
				dsPointer+=1
	def setup(self,folder,mode): #opens and sets up csvs
		if mode == 0:
			csvLst = os.scandir(folder) #opens data folder
			data = [pd.read_csv(review) for review in csvLst] #reads the csvs into an array as dataframes
			pickle.dump(data, open( "csvArray.p", "wb" ) )
			return data
		if mode == 1:
			print("Import CSV Loaded \n")
			return pickle.load( open( "csvArray.p", "rb" ) )
		if mode > 1:
			print("Import CSV Skipped \n")
			return None

	def process(self,dataArrayRaw,mode):#converts csvs to python list for easier ann training
		if mode < 2: #if we need to make a new data array to use for training 
			dataArray = []
			for x in range(len(dataArrayRaw)):#create an array in which each index is an array of 2 arrays (x, y's)
				dataArray.append([[],[]]) 
			# print(dataArray)
			for x in range(len(dataArrayRaw)): #loop through each of our datasets
				dataArrayRaw[x] = dataArrayRaw[x].drop(["TOTAL","PHRASE"],axis=1)#drop non data columns
				#print(dataArrayRaw[x])
				for y in range(0,dataArrayRaw[x].shape[0]):#add the x vals and the corrisponding source distributions for each x
					dataArray[x][0].append(y)
					dataArray[x][1].append(list(dataArrayRaw[x].loc[y]))
				# print(dataArray[x][0][0],dataArray[x][0][1])
			# print(dataArray)
			pickle.dump(dataArray, open( "processedData.p", "wb" ) )
			return dataArray
		if mode == 2:
			print("CSV Processing Loaded \n")
			return pickle.load( open( "processedData.p", "rb" ) )
		if mode > 2: 
			print("CSV Processing Skipped \n")
			return None

	def buildNetworks(self,dataArray,mode,dsIndex=None): #builds a blank ANN for each data set
		#ds is an array that holds n 
		#creates an ann of input 1 (the word/ col) and output of len i (word column of news sources appearences)
		if dsIndex == None:
			dummyModelList = [self.neural_network_model(1, (len(ds[1][0]))) for ds in dataArray]
			#pickle.dump(dummyModelList, open("dummyModelList.p", "wb" ) )
		else:
			dummyModelList = []
			for x in range(len(dataArray)):
				if x in dsIndex:
					dummyModelList.append(self.neural_network_model(1, (len(dataArray[x][1][0]))))
			#pickle.dump(dummyModelList, open("dummyModelList.p", "wb" ) )
		return dummyModelList

	def neural_network_model(self, input_size, output_size): #generates the actual ANN
	    # print("Building Model...\n")
	    LR = 1e-3

	    network = tf.keras.models.Sequential()
	    network = input_data(shape=(1,1), name='input') #takes in the board state as an input
		
	    network = fully_connected(network, 128, activation='relu') #hidden ANN layer 1 all hidden layers use relu
	    network = dropout(network, 0.8) #dropout 80% of all data at a given level to decrease flexibilty and increase generalization
	    # network = dropout(network, 0.95)
	
	    network = fully_connected(network, 256, activation='relu') #hidden ANN layer 2
	    network = dropout(network, 0.8)
	    # network = dropout(network, 0.95)
	
	    network = fully_connected(network, 512, activation='relu') #hidden ANN layer 3
	    network = dropout(network, 0.8)
	    # network = dropout(network, 0.95)
	
	    network = fully_connected(network, 512, activation='relu') #hidden ANN layer 4
	    network = dropout(network, 0.8)
	    # network = dropout(network, 0.95)
	
	    network = fully_connected(network, 256, activation='relu') #hidden ANN layer 5
	    network = dropout(network, 0.8)
	    # network = dropout(network, 0.95)
	
	    network = fully_connected(network, 128, activation='relu') #hidden ANN layer 6
	    network = dropout(network, 0.8)
	    #network = dropout(network, 0.95)
	
	    network = fully_connected(network, output_size, activation='softmax') #prints out an output matrix of probability for each given move
	    # network = fully_connected(network, gamesize**2, activation='relu')
	
	    #defines methods of calculus to use for regression
	    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
	    model = tflearn.DNN(network, tensorboard_dir='log')
	    print(output_size)
	    return model

	def trainNetworks(self,dummyModelList,dataArray,dsIndex,mode):
		#need a way to only access the parts of dataArray that corrispond with what exists in dummy model list
		#sudo: dummyList[x].fit(data[f(x)][0],data[f(x)][1]) 
		sess = tf.compat.v1.Session()
		saver = tf.compat.v1.train.Saver()
		with sess:
			if mode < 3:
				dsPointer = 0 #allows us to increment through data array and dummyModelListSeperatly
				#this is important because dataArray is len n wheras dummyModelList is len dsIndex
				for x in range(len(dataArray)):
					if x in dsIndex:
						# print(dataArray[x][0])
						y = np.array([np.array(dataPoint).reshape(len(dataPoint)) for dataPoint in dataArray[x][1]])
						print(y.shape)
						print(dummyModelList[dsPointer])
						print(x)
						dummyModelList[dsPointer].fit({'input': np.array(dataArray[x][0]).reshape(len(dataArray[x][0]),1,1)}, {'targets': y}, n_epoch=10, snapshot_step=500, show_metric=True, run_id='openai_learning')
						dsPointer+=1
				# for x in dsIndex:
					# saver.save(sess, "./model_"+str(x))
				# print (dummyModelList)
				return dummyModelList
			if mode == 3:
				print("ANN Training Loaded \n")
				for x in dsIndex:
					modelList = [saver.restore(sess, "./model_"+str(x))]
				return 
			if mode > 3:
				print("ANN Training Skipped \n")
				return None

DataProcessor("./phrasebias_data/phrase_counts",2,[1])
		
		