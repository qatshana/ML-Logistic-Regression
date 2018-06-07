'''
891 samples of titanic passengers for Kaggle 

Features include ASex, Age, Cabin and others

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Load dataset from csv file
train = pd.read_csv('titanic_train.csv')

# get basic information about the dataset
train.head()
train.describe()
train.info()

# get dataset columns
train.columns

# explore data - relationship between different variables

#find missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Count survival rate total

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

# Count survival rate by gender
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

# Count survival rate by class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

# get distribution by age
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

# or use dataframe to plot distribution results
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

# draw distibution of number of siblings
sns.countplot(x='SibSp',data=train)

# get distribution of fair using pandas
train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# Data Cleaning
# plot age by class and derive age by class so we can replace missing data
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

# create function to replace missing data with average
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

# call function and apply to replace null values
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#plot results to confirm that all missing points have been replaced
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#drop Cabin column as it has a lot of missing data that can not be replaced
train.drop('Cabin',axis=1,inplace=True)
# drop not a number
train.dropna(inplace=True)

# Converting Categorical Features 
# convert categorial features into dummy variables

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)


#Training a Linear Regression Mode

# define inputs (ignore address)
X=train.drop('Survived',axis=1)

#define output file
y=train['Survived']

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