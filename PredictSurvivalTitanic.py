'''

Machine Learning Program to Predict Survival in Titanic Ship

Program domestrate how to train  Logistical Regression to predict outcomes 

Data        ---  891 samples of titanic passengers from Kaggle dataset (data set has 11 features) 

Features    ---  include Sex, Age, Cabin and others

Target      ---  Survised (1), did not Survive (0)

Performance ---  Classifier acheives accuracy of 82%


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

def load_training_data():
    '''
    Load training data from csv file
    output -- training set in dataframe format

    '''
    train = pd.read_csv('titanic_train.csv')
    return train

def explore_data(train):
    '''
    Explore data - relationship between different variables

    '''
    # get basic information about the dataset
    train.head()
    train.describe()
    train.info()

    # get dataset columns
    train.columns 

    #find missing data
    sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

    # Count survival rate total
    sns.set_style('whitegrid')
    sns.countplot(x='Survived',data=train,palette='RdBu_r')

    # Count survival rate by gender
    sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
    plt.show()

    # Count survival rate by class
    sns.set_style('whitegrid')
    sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
    plt.show()

    # get distribution by age
    sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
    plt.show()

    # or use dataframe to plot distribution results
    train['Age'].hist(bins=30,color='darkred',alpha=0.7)
    plt.show()

    # draw distibution of number of siblings
    sns.countplot(x='SibSp',data=train)
    plt.show()


    # get distribution of fair using pandas
    train['Fare'].hist(color='green',bins=40,figsize=(8,4))
    plt.show()

    # Data Cleaning
    # plot age by class and derive age by class so we can replace missing data
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


def impute_age(cols):
    '''
     Replace missing age data with average value for the class
    
    '''
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

def process_data(train):
    '''
    Process data, replace missing age values with average value
    '''
    train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)     # call function and apply to replace null values
    sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')     #plot results to confirm that all missing points have been replaced
    plt.show()  # confirm that age missing data was replaced successfully
    train.drop('Cabin',axis=1,inplace=True)   #drop Cabin column as it has a lot of missing data that can not be replaced 
    train.dropna(inplace=True)   # drop any other missing data
    sex = pd.get_dummies(train['Sex'],drop_first=True)     # Converting Categorical Features    
    embark = pd.get_dummies(train['Embarked'],drop_first=True) # convert categorial features into dummy variables
    train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
    train = pd.concat([train,sex,embark],axis=1) # re-add all columns except Ticket number as it does not add value 
    return train

def split_feature_target(train):
    '''
    Split feature vs. target 

    '''
    X=train.drop('Survived',axis=1) # define features
    y=train['Survived']     # define taget 
    return X,y    

if __name__=='__main__':

    train=load_training_data() # load data
    train=process_data(train)  # process/clean data

    X,y=split_feature_target(train)

    #Create train and est sets using Train Test Split, 40% test data and 60% training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    #Training a Linear Regression Mode
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)        # Train Model    
   
    predictions = logmodel.predict(X_test)  # Perform predictions

    # Model Evaluation - 
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

