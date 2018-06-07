'''
1000 samples of customers info 


'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load dataset from csv file
train = pd.read_csv('advertising.csv')

# get basic information about the dataset
train.head()
train.describe()
train.info()

# get dataset columns
train.columns

# explore data - relationship between different variables


# get distribution by age
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

# or use dataframe to plot distribution results
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

# jointplot
sns.jointplot(x=train['Age'],y=train['Area Income'])
sns.jointplot(x=train['Age'],y=train['Daily Time Spent on Site'])

sns.jointplot(x=train['Daily Time Spent on Site'],y=train['Daily Internet Usage'])


sns.pairplot(train)

#Training a Linear Regression Mode

# define inputs (ignore address)
train.drop(['Country','Ad Topic Line','City','Timestamp'],axis=1,inplace=True)

X=train.drop('Clicked on Ad',axis=1)

#define output file
y=train['Clicked on Ad']

#Create train and est sets using Train Test Split, 40% test data and 60% training data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train Model
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Perform predictions

predictions = logmodel.predict(X_test)

# Model Evaluation - 

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))