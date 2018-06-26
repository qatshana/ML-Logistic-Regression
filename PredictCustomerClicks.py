
'''

Machine Learning Program to Predict Survival in Titanic Ship

Program domestrate how to train  Logistical Regression to predict outcomes 

Data        ---  1000 samples of customers info (data set has 9 features) 

Features    ---  include Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Ad Topic Line, City, Male, Country, Timestamp

Target      ---  Clicked on Add (1)

Performance ---  Classifier acheives accuracy of 92%


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_training_data():
    '''
    Load training data from csv file
    output -- training set in dataframe format

    '''
    train = pd.read_csv('advertising.csv')
    return train


def explore_data(train):
	'''
    Explore data - relationship between different variables

    '''
	train.head()
	train.describe()
	train.info()
	train.columns 	# get dataset columns
	
	sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30) # get distribution by age
	train['Age'].hist(bins=30,color='darkred',alpha=0.7) # or use dataframe to plot distribution results

	# jointplot
	sns.jointplot(x=train['Age'],y=train['Area Income'])
	sns.jointplot(x=train['Age'],y=train['Daily Time Spent on Site'])
	sns.jointplot(x=train['Daily Time Spent on Site'],y=train['Daily Internet Usage'])
	sns.pairplot(train)

def process_data(train):
	'''
    Process data, replace missing age values with average value
    '''
	train.drop(['Country','Ad Topic Line','City','Timestamp'],axis=1,inplace=True)  # define inputs (ignore address info and ad topic/timestamp
	return train


def split_feature_target(train):
	'''
    Split feature vs. target 
        
    '''
	X=train.drop('Clicked on Ad',axis=1) # define inputs (ignore address)
	y=train['Clicked on Ad'] # define output file
	return X,y  

if __name__=='__main__':
	train=load_training_data() # load data
	train=process_data(train)  # process/clean data
	X,y=split_feature_target(train)

	#Create train and est sets using Train Test Split, 30% test data and 70% training data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

	#Training a Linear Regression Mode
	logmodel = LogisticRegression()
	logmodel.fit(X_train,y_train)

	# Perform predictions
	predictions = logmodel.predict(X_test)

	# Model Evaluation - 
	print(classification_report(y_test,predictions))