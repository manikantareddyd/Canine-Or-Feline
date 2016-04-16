from sklearn.svm import *
from data import *
from sklearn.decomposition import RandomizedPCA
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import * #Contains necessary kernels except intersection
class classifier:
	def __init__(self,Data,gamma=None,n_estimators=None):
		self.load(Data,gamma,n_estimators=n_estimators)
	def intersectionKernel(self,X1,X2):
	    K=np.zeros((X1.shape[0],X2.shape[0]))
	    for i in range(len(X1)):
	    	for j in range(len(X2)):
	    		v1=X1[i]
	    		v2=X2[j]
	    		val=0
	    		for k in range(len(v1)):
	    			val += min([v1[k],v2[k]])
	    		K[i][j] = val
	    return K
	def load(self,Data,gamma=None, n_estimators=None):
		xTrain = np.array(Data.trainingData[:2000],dtype='float32')
		yTrain = Data.trainingDataLabels[:2000]
		xTest = np.array(Data.testingData[:500],dtype='float32')
		yTest = Data.testingDataLabels[:500]
		# print "Selecting Kernel"
		# selected = laplacian_kernel(gamma=gamma)
		# selected = intersectionKernel
		# selected = 'rbf'
		# selected = 'poly'
		# selected = 'linear'
		# print "Building Model"
		# clf = SVC(kernel=selected,C=0.0001)
		# clf = LinearSVC()
		clf = RandomForestClassifier(n_estimators = n_estimators)
		clf.fit(xTrain,yTrain)
		# print "Model Built"
		yPred = clf.predict(xTest)
		# print clf.score(xTest,yTest)
		print classification_report(yTest, yPred)

for i in range(30,230,10):
	classifier(a,n_estimators=i)

# for i in [0.1, 0.002, 0.0001, 1]:
# 	classifier(a)
