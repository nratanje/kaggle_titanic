import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn as skl
from sklearn import preprocessing
#from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

use_train_for_testing = True

proportion_for_testing = 0.66

#===================================================================

kaggle_titanic_main_dir = os.path.dirname(__file__)
train_csv_file          = open(kaggle_titanic_main_dir + "/train.csv", 'r')
train_dataframe         = pd.read_csv(train_csv_file).replace(np.nan, -1)

#===================================================================

#["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

train_dataframe = train_dataframe.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], 1)
train_dataframe = train_dataframe[["Survived","Pclass","Sex","Age","Fare","PassengerId"]]

#===================================================================

#Replace string entries with numbers

train_dataframe["Sex"] = train_dataframe["Sex"].map({'male': 0, 'female': 1})

#-------------------------------------------------------------------

if use_train_for_testing == True:

	no_rows_for_testing = int(round((1-proportion_for_testing)*len(train_dataframe),0))
	
	test_dataframe  = train_dataframe[no_rows_for_testing+1:]
	train_dataframe = train_dataframe[:no_rows_for_testing]
	
	print "training with " + str(len(train_dataframe)) + " rows"
	print "testing with "  + str(len(test_dataframe))  + " rows"; print
	
if use_train_for_testing == False:
	test_csv_file  = open(kaggle_titanic_main_dir + "/test.csv", 'r')
	test_dataframe = pd.read_csv(test_csv_file).replace(np.nan, -1)
	test_dataframe = test_dataframe.drop(["Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], 1)
	test_dataframe = test_dataframe[["Pclass","Sex","Age","Fare","PassengerId"]]
	test_dataframe["Sex"] = test_dataframe["Sex"].map({'male': 0, 'female': 1})

test_nparray  = test_dataframe.as_matrix()
train_nparray = train_dataframe.as_matrix()

passenger_training = train_nparray[:, [1,2,3,4]]
survived_training  = train_nparray[:, 0]

#-------------------------------------------------------------------

if use_train_for_testing == True:
	passenger_testing = test_nparray[:, [1,2,3,4]]
	survived_testing  = test_nparray[:, 0]

	skl_dtclf = RandomForestClassifier(n_estimators=500, min_samples_split=8, n_jobs=-1, random_state=1)
	skl_dtclf = skl_dtclf.fit(passenger_training, survived_training)

	skl_predictions = skl_dtclf.predict(passenger_testing)

	def percent_diff(list_A,list_B):

		if len(list_A) != len(list_B):
			return 0
			
		else:
			number_of_same = 0
			for i in range(0,len(list_A)):
				if list_A[i] == list_B[i]:
					number_of_same += 1
			return number_of_same/float(len(list_A))

	print percent_diff(skl_predictions, survived_testing)

if use_train_for_testing == False:
	passenger_testing = test_nparray[:, [0,1,2,3]]

	#print passenger_testing
	
	skl_dtclf = RandomForestClassifier(n_estimators=500,min_samples_split=8, n_jobs=-1, random_state=1)
	skl_dtclf = skl_dtclf.fit(passenger_training, survived_training)

	skl_predictions = skl_dtclf.predict(passenger_testing)
	
	predicted_survived         = pd.DataFrame(skl_predictions)
	predicted_survived.columns = ["Survived"]
	
	predicted_survived["Survived"] = pd.to_numeric(predicted_survived["Survived"], errors=-1, downcast='unsigned')
	
	test_dataframe["PassengerId"]
	
	predicted_results = pd.concat([test_dataframe["PassengerId"], predicted_survived["Survived"]], axis=1)

	predicted_results.to_csv(kaggle_titanic_main_dir + "/predicted_results.csv", index=0)
	
print; print "fin"