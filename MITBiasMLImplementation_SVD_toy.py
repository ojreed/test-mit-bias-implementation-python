"""
run code with $py -3.8 MITBiasMLImplementation.py


"""


import numpy as np
import pandas as pd
import re
import nltk
import pickle
import os
import argparse
import random
from statistics import median, mean
from collections import Counter
from numpy.random import choice
import scipy
import matplotlib.pyplot as plt
from numpy import random


class DataProcessor(object):
	def __init__(self, folder,mode,targetDS=None):
		"""
		Mode: 
			allows for different reference data to be pulled from pickel rather than processed
			0 - runs all processing from beggining 
			1 - skips to loading csv files
			2 - skips to loading process 
			3 - skips to loading results array

		"""
		self.mode = mode 
		self.targetDS = targetDS
		self.folder = folder
		self.r = 3
		#data setup
		self.data_init()
		#run code
		self.main()

	def main(self):
		self.outputs = self.optimize_handeler(self.mode)
		self.display_datasets()
		"""
		TODO:
			impliment printout function
				turns cols 1 and 2 of U and V into datapoints 
		"""

	def data_init(self):
		"""
		TODO:
			modify original data storage to store list of news sources and list of terms
				store this data
		"""
		self.dataArrayRaw = self.setup(self.folder,self.mode) 
		self.dataArray = self.process(self.dataArrayRaw,self.mode)
		if self.targetDS == None:
			self.targetDS = [x for x in range(len(self.dataArray))]

	def setup(self,folder,mode): #opens and sets up csvs
		if mode == 0:
			print("Import CSV Computing \n")
			csvLst = os.scandir(folder) #opens data folder
			data = [pd.read_csv(review) for review in csvLst] #reads the csvs into an array as dataframes
			pickle.dump(data, open( "csvArray_T.p", "wb" ) )
			return data
		if mode == 1:
			print("Import CSV Loaded \n")
			return pickle.load( open( "csvArray_T.p", "rb" ) )
		if mode > 1:
			print("Import CSV Skipped \n")
			return None

	def process(self,dataArrayRaw,mode):#converts csvs to python list for easier ann training
		if mode < 2: #if we need to make a new data array to use for training 
			print("CSV Processing Computing\n")
			dataArray = []
			# print(dataArray)
			for x in range(len(dataArrayRaw)): #loop through each of our datasets
				tempP = np.array(dataArrayRaw[x]["PHRASE"])
				tempN = np.array(dataArrayRaw[x].drop(["TOTAL","PHRASE"],axis=1))#drop non data columns
				tempS = np.array(dataArrayRaw[x].columns)[2:]
				# print(tempN)
				# print(tempP)
				# print(tempS)	
				dataArray.append((tempN,tempP,tempS))
			pickle.dump(dataArray, open( "processedData_T.p", "wb" ) )
			return dataArray
		if mode >= 2:
			print("CSV Processing Loaded \n")
			return pickle.load( open( "processedData_T.p", "rb" ) )
		# if mode > 2: 
		# 	print("CSV Processing Skipped \n")
		# 	return None

	"""
	takes a numerical input 
	returns the ReLU of it
	"""
	def SVD_ReLU(self,inp):
		if inp >= 0:
			return inp
		else:
			return None

	#takes a 1d flattened index and converts to a 2d index
	def convert_1d_2d(self,rows,cols,idx):
		row_idx = idx//cols
		col_idx = idx%cols
		return row_idx, col_idx

	#flattens a 2d index into a 1d index
	def convert_2d_1d(self,cols,idx_r,idx_c):
		return idx_r*cols + idx_c

	"""
	takes r value, index i and j, then W U and V matricies 
	returns N_bar[i][j]
	"""
	def idx_n_bar_calc(self,r, i, j, W, U, V):
		raw_n_bar = 0
		for k in range(r):
			raw_n_bar+= W[k]*U[i][k]*V[j][k]
		return self.SVD_ReLU(raw_n_bar)

	def parse_inputs(self,inp):
		"""
		first r inputs 
			W matrix 
		next N
		"""
		N = self.N
		r = self.r #initialize r to 3 because of MIT article
		n = (self.N.shape)[1] #n = cols --> corrisponds with i/media source
		m = (self.N.shape)[0] #m = rows --> corrisponds with j/phrase 
		W = np.zeros(r)
		U = np.zeros([n,r])
		V = np.zeros([m,r])
		for idx in range(len(inp)):
			if idx <r:
				W[idx] = inp[idx]
			elif idx<r+n*r:
				idx_r, idx_c = self.convert_1d_2d(n,r,idx-r)#ignores the first r indicies meant for W 
				U[idx_r][idx_c] = inp[idx]
			else:
				idx_r, idx_c = self.convert_1d_2d(m,r,idx-r-n*r)#ignores the first r + n*r indicies meant for W and U
				V[idx_r][idx_c] = inp[idx]
		return W,U,V

	def rosen_der(self,x):
		xm = x[1:-1]
		xm_m1 = x[:-2]
		xm_p1 = x[2:]
		der = np.zeros_like(x)
		der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
		der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
		der[-1] = 200*(x[-1]-x[-2]**2)
		return der
	def rosen_hess(self,x):
	    x = np.asarray(x)
	    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
	    diagonal = np.zeros_like(x)
	    diagonal[0] = 1200*x[0]**2-400*x[1]+2
	    diagonal[-1] = 200
	    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
	    H = H + np.diag(diagonal)
	    return H

	"""
	takes input of initialized W U V and original N matrix
	returns optimized U and V matricies 

	W = len r array 
	U = n rows r cols
	V = m rows r cols
	N = m rows n cols

	"""
	def LnLpoi(self,inp):
		W,U,V = self.parse_inputs(inp)
		# print(U)
		N = self.N
		r = self.r #initialize r to 3 because of MIT article
		n = (self.N.shape)[1] #n = cols --> corrisponds with i/media source
		m = (self.N.shape)[0] #m = rows --> corrisponds with j/phrase 
		N_bar = np.zeros([m,n])
		LnLpoi_val = 0
		for i in range(n):
			for j in range(m):
				try:
					N_bar[j][i] = self.idx_n_bar_calc(r,i,j,W,U,V)
					# if (N_bar[j][i] == None):
					# 	# print(N_bar[j][i])
					# 	# print("HIT")
					# 	return None
					LnLpoi_val_log = np.log(N[j][i]/(np.e*N_bar[j][i]))
					if np.isinf(LnLpoi_val_log):
					# if (LnLpoi_val_log< -100000000):
						# print(LnLpoi_val_log)
						# print(N[j][i],N_bar[j][i])
						# print(N)
						# print(U[i])
						# print(V[j])
						# print(W)
						# exit()
						# print("hit")
						LnLpoi_val += N_bar[j][i]
					else:	
						LnLpoi_val += (N_bar[j][i] + N[j][i]*np.log(N[j][i]/(np.e*N_bar[j][i])))
				except:
					LnLpoi_val += 10000
				# print(np.log(N[j][i]/(np.e*N_bar[j][i])))
				# print(LnLpoi_val)
				# print(N_bar[j][i])
		# print(LnLpoi_val)
		# exit()
		# print(N_bar[10][5],N[10][5])
		# print(W)
		# print(N_bar,N)
		# print("\n")
		# print(LnLpoi_val)
		# print("\n")
		return LnLpoi_val 

	def optimize_handeler(self,mode):
		if mode <3:
			print("SVD Processing Computing \n")
			# outputs = [[] for x in range(len(self.dataArray))]
			outputs = pickle.load( open( "SVD_Outputs_diff_ev_T.p", "rb" ) )
			for DS in self.targetDS:
				self.N = self.dataArray[DS][0]
				# print(self.N.shape)
				n = (self.N.shape)[1] #n = cols --> corrisponds with i/media source
				m = (self.N.shape)[0] #m = rows --> corrisponds with j/phrase
				r = self.r
				# inp = np.random.rand(r + n*r + m*r)
				inp = np.ones(r + n*r + m*r)
				# for index in range(len(inp)):
					# inp[index] = random.randint(500)
					# inp[index] = 50
				# ,jac=self.rosen_der, hess=self.rosen_hess
				# out = scipy.optimize.minimize(self.LnLpoi,inp,method='Nelder-Mead',options={'xatol':1, 'maxiter':1000*len(inp),'disp': True})
				out = scipy.optimize.differential_evolution(self.LnLpoi,[[0.1,np.sqrt(np.amax(self.N))] for x in range(len(inp))],disp = True, polish = True, updating = 'immediate',recombination = .75)
				# print(out)
				W,U,V = self.parse_inputs(out.x)
				print("U Matrix of "+ str(DS) + ":\n")
				print(U)
				print("\nV Matrix of " + str(DS) + ":\n")
				print(V)
				outputs[DS] = [W,U,V]
			pickle.dump(outputs, open( "SVD_Outputs_diff_ev_T.p", "wb" ) )
			return outputs
		if mode == 3:
			print("SVD Processing Loaded \n")
			# self.N = self.dataArray[20][0]
			return pickle.load( open( "SVD_Outputs_diff_ev_T.p", "rb" ) )
		if mode > 3: 
			print("SVD Processing Skipped \n")
			# self.N = self.dataArray[20][0]
			return None
	def display_datasets(self):
		for DS in self.targetDS:
			U = self.outputs[DS][1]
			V = self.outputs[DS][2]
			W = self.outputs[DS][0]
			# print(self.idx_n_bar_calc(3, 0, 0, W, U, V))	
			# print(self.N[0,0])		
			#plot news sources
			#Best Abortion 0 2
			y = U[:,0]
			x = U[:,2]
			n = self.dataArray[DS][2]
			fig, ax = plt.subplots()
			ax.scatter(x, y)
			for i, txt in enumerate(n):
				ax.annotate(txt, (x[i], y[i]))
			plt.title("Topic" + str(DS))
			plt.show()
			#plot topics
			y = V[:,0]
			x = V[:,2]
			n = self.dataArray[DS][1]
			fig, ax = plt.subplots()
			ax.scatter(x, y)
			for i, txt in enumerate(n):
				ax.annotate(txt, (x[i], y[i]))
			plt.title("Topic" + str(DS))
			plt.show()
	




DataProcessor("./phrasebias_data/toy_data",3,[1])
		
		