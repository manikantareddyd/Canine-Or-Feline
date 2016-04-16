from sklearn.svm import *
from data import *
import random
from sklearn.decomposition import RandomizedPCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import * #Contains necessary kernels except intersection
from sklearn.metrics import classification_report
class tiger_leopard:
	def __init__(self,Data, n_estimators=None):
		tigers = Data.felines['tiger']
		leopards = Data.felines['leopard']
		tup = [(i,'tiger') for i in tigers]+[(i,'leopard') for i in leopards]
		random.shuffle(tup)
		X = [tup[i][0] for i in range(len(tup))]
		y = [tup[i][1] for i in range(len(tup))]
		# X = tigers + leopards
		# y = ['tiger' for i in tigers]+['leopard' for i in leopards]
		xTrain, xTest, yTrain, yTest = train_test_split(
				X,
				y,
				test_size = 0.25,
				random_state = datetime.now().second
				)
		# clf = RandomForestClassifier(n_estimators=n_estimators)
		clf = LinearSVC()
		clf.fit(xTrain,yTrain)
		yPred = clf.predict(xTest)
		print clf.score(xTest,yTest)
		print classification_report(yTest, yPred)

class AllForest:
	def __init__(self,Data, n_estimators=500):
		tups = []
		for i in Data.felines:
			for vec in Data.felines[i]:
				tups.append((i,vec))
		for i in Data.canines:
			for vec in Data.canines[i]:
				tups.append((i,vec))
		random.shuffle(tups)
		X = [tups[i][1] for i in range(len(tups))]
		y = [tups[i][0] for i in range(len(tups))]
		xTrain, xTest, yTrain, yTest = train_test_split(
				X,
				y,
				test_size = 0.25,
				random_state = datetime.now().second
				)
		clf = RandomForestClassifier(n_estimators=n_estimators)
		# clf = LinearSVC()
		clf.fit(xTrain,yTrain)
		yPred = clf.predict(xTest)
		# print clf.score(xTest,yTest)
		print classification_report(yTest, yPred)

# c= tiger_leopard(a)

c = AllForest(a)