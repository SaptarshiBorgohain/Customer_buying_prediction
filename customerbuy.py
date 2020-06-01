#Do this in jupyter notebook for better results!
#In cell 1
#Import all that you need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

#In cell 2
#Download the file in your device and upload the file to jupyter 

a = pd.read_csv('customerbuy2 - customerbuy2.csv.csv')
a

#In cell 3
#spliting the data base
x = a.iloc[:,:-1]
y = a.iloc[:,-1]
x

#In cell 4
#Now from sklearn we import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=0)

#from Sklearn we import DecisionTree
#IN cell 5
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(x_train, y_train)

#In cell 6
#Run the cell to find the predicted values
y_pred = classifier.predict(x_test)
y_pred

#cell 7
#test
y_test

#In cell 8
#we are ready for new prediction.
new_prediction = classifier.predict(np.array([[61,70000],[65,90000],[27,50000]]))
new_prediction

#happy prediction.