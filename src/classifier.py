from sklearn.svm import *
from data import *

class classifier:
	def __init__(self,Data):
		self.load(Data)

	def load(self,Data):
		clf = SVC(kernel='intersection')
		print "Building Model"
		clf.fit(Data.trainingData,Data.trainingDataLabels)
		print "Model Built"
		print clf.score(Data.testingData,Data.testingDataLabels)

c= classifier(a)