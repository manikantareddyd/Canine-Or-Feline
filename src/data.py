from os import listdir
import numpy as np
from sklearn.cross_validation import train_test_split
from datetime import datetime
import random

class Data:
	def __init__(self):
		self.canines, self.felines = self.load()
		self.generateDatasets()
	def load(self):
		path="../datasets/"
		canines = {}
		felines = {}
		dogs = listdir(path+'canine')
		cats = listdir(path+'feline')

		for dog in dogs:
			name = dog[:-4]
			canines[name]=[]
			f=open(path+'canine/'+dog,'r')
			rawTxt = f.read()
			f.close()
			splitlines = rawTxt.split('\n')[:-1]
			for line in splitlines:
				vector = line.split(' ')
				vector = np.array([float(vector[i]) for i in range(len(vector))])
				vector = vector / np.linalg.norm(vector,2)
				canines[name].append(vector)

		for cat in cats:
			name = cat[:-4]
			felines[name]=[]
			f=open(path+'feline/'+cat,'r')
			rawTxt = f.read()
			f.close()
			splitlines = rawTxt.split('\n')[:-1]
			for line in splitlines:
				vector = line.split(' ')
				vector = np.array([float(vector[i]) for i in range(len(vector))])
				vector = vector / np.linalg.norm(vector,2)
				felines[name].append(vector)

		return canines, felines

	def generateDatasets(self):
		vectors = []
		
		self.trainingData = []
		self.trainingDataLabels = []
		trainingTuples = []
		
		self.testingData = []
		self.testingDataLabels = []
		testingTuples = []
		
		for dog in self.canines:
			xTrain, xTest, yTrain, yTest = train_test_split(
				self.canines[dog],
				[1 for i in range(len(self.canines[dog]))],
				test_size = 0.25,
				random_state = datetime.now().second
				)
			[trainingTuples.append((xTrain[i],yTrain[i])) for i in range(len(xTrain))]
			[testingTuples.append((xTest[i],yTest[i])) for i in range(len(xTest))]

		for cat in self.felines:
			xTrain, xTest, yTrain, yTest = train_test_split(
				self.felines[cat],
				[0 for i in range(len(self.felines[cat]))],
				test_size = 0.25,
				random_state = datetime.now().second
				)
			[trainingTuples.append((xTrain[i],yTrain[i])) for i in range(len(xTrain))]
			[testingTuples.append((xTest[i],yTest[i])) for i in range(len(xTest))]
		
		random.seed = datetime.now().second
		random.shuffle(trainingTuples)
		random.shuffle(testingTuples)
		
		for i in range(len(trainingTuples)):
			self.trainingDataLabels.append(trainingTuples[i][1])
			self.trainingData.append(trainingTuples[i][0])
		
		for i in range(len(testingTuples)):
			self.testingDataLabels.append(testingTuples[i][1])
			self.testingData.append(testingTuples[i][0])

		# self.trainingData = np.array(self.trainingData,dtype='float32')
a=Data()
print "Data Loaded"