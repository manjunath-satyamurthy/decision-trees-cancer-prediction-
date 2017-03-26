from sklearn import tree
import pydotplus
import numpy as np
import sys

input_data = open("training_sample.csv", "r").read().split("\n")


features = input_data[0].split(",")
del features[3] # removing cancerL label
del features[3] # removing cancerR label

del input_data[0] # removing csv header
del input_data[-1] # removing last empty line

class_names = ["No cancer", "Cancer Left Breast", "Cancer Right Breast", "Cancer Both Breasts"]
samples = []
sample_classes = []

def cancer_enum(cancerL, cancerR):
	if cancerL == 1 and cancerR == 1: # creating a list of classes to which respective samples belong to
		return 3 # 3 meaning cancer on both left and right breast
	elif cancerL == 1 and cancerR == 0:
		return 1 # 1 meaning cancer on left brest only
	elif cancerL == 0 and cancerR == 1:
		return 2 # 2 meaning cancer on right breast only
	elif cancerL == 0 and cancerR == 0:
		return 0 # 0 meaning, No cancer on either breasts
	else:
		return 0


for i, line in enumerate(input_data):
	data = line.split(",")
	cancerL, cancerR = int(data[3]), int(data[4])
	del data[3] # removing cancerL value from array
	del data[3] # removing cancerR value from array
	samples.append(data)
	sample_classes.append(cancer_enum(cancerL, cancerR))


clf = tree.DecisionTreeClassifier()
clf = clf.fit(samples, sample_classes)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features,
	class_names=class_names, filled=True, rounded=True,	special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("classification.pdf") # generates tree classification as a pdf

test_data = open("test_sample.csv", "r").read().split("\n")
del test_data[0] # removing csv header
del test_data[-1]

samples_count = 0
success = 0
for i, line in enumerate(test_data): #testing the classification accuracy
	data = line.split(",")
	cancerL, cancerR = int(data[3]), int(data[4])
	del data[3] # removing cancerL value from array
	del data[3] # removing cancerR value from array
	expected_result = cancer_enum(cancerL, cancerR)
	samples_count += 1
	result = clf.predict(np.array(data).reshape(1,-1))[0]
	if result == expected_result:
		success += 1

print "\nclassification accuracy = "+str((100*success)/samples_count)+"%\n"

print "To view the classification, open \"classification.pdf\" that has been generated. \n"


def predict_cancer():
	print "enter the following details in integers only and enter -99 for unknown attributes\n\n"

	subjectId = raw_input("subjectId\n")
	examIndex = raw_input("examIndex\n")
	daysSincePreviousExam = raw_input("daysSincePreviousExam\n")
	invL = raw_input("invL\n")
	invR = raw_input("invR\n")
	age = raw_input("age\n")
	implantEver = raw_input("implantEver\n")
	implantNow = raw_input("implantNow\n")
	bcHistory = raw_input("bcHistory\n")
	yearsSincePreviousBc = raw_input("yearsSincePreviousBc\n")
	previousBcLaterality = raw_input("previousBcLaterality\n")
	reduxHistory = raw_input("reduxHistory\n")
	reduxLaterality = raw_input("reduxLaterality\n")
	hrt = raw_input("hrt\n")
	antiestrogen = raw_input("antiestrogen\n")
	firstDegreeWithBc = raw_input("firstDegreeWithBc\n")
	firstDegreeWithBc50 = raw_input("firstDegreeWithBc50\n")
	bmi = raw_input("bmi\n")
	race = raw_input("race\n\n\n")

	input_data = np.array([subjectId, examIndex, daysSincePreviousExam,
		invL, invR, age, implantEver, implantNow, bcHistory, yearsSincePreviousBc,
		previousBcLaterality, reduxHistory, reduxLaterality, hrt, antiestrogen, 
		firstDegreeWithBc, firstDegreeWithBc50, bmi, race]).reshape(1, -1)

	result = clf.predict(input_data)[0]
	print "Cancer Prediction :\n"
	if result == 0:
		print "cancerL = 0, cancerR = 0\n\n\n"
	elif result == 1:
		print "cancerL = 1, cancerR = 0\n\n\n"
	elif result == 2:
		print "cancerL = 0, cancerR = 1\n\n\n"
	elif result == 3:
		print "cancerL = 1, cancerR = 1\n\n\n"

while True:
	option = raw_input("press y to predict and n to exit\n")

	if option == 'y' or option == "Y":
		predict_cancer()
	elif option =='n'or option == 'N':
		sys.exit()

