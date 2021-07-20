<h1>SVM CODE:</h1>

<h3>CODE BY 64359-SHAAN</h3>

import pandas as pd

import numpy  as np

np.random.seed(2018)

feature_names = ['Age','Pclass','Embarked','Sex']

# loading data sets 

train = pd.read_csv('train.csv', usecols =['Survived','PassengerId'] + feature_names)

test  = pd.read_csv('test.csv',  usecols =['PassengerId'] + feature_names )

# combining the train and test file for joint processing purposes 

test['Survived'] = np.nan

comb = pd.concat([ train, test ])

comb.head()

print('Number of missing Embarked values ',comb['Embarked'].isnull().sum())

comb['Embarked'] = comb['Embarked'].fillna('S')

comb['Embarked'].unique()

comb['NoAge'] = comb['Age'] == np.NAN

comb['Age'] =  comb['Age'].fillna(-1)

comb['Age'].hist(bins=100)

comb['Minor'] = (comb['Age']<14.0)&(comb['Age']>=0)

# Pclass one hot encode
comb['P1'] = comb['Pclass'] == 1 

comb['P2'] = comb['Pclass'] == 2

comb['P3'] = comb['Pclass'] == 3

# Embarked one hot encode

comb['ES'] = comb['Embarked'] == 'S' 

comb['EQ'] = comb['Embarked'] == 'Q'

comb['EC'] = comb['Embarked'] == 'C'

# encode Sex
comb['Sex'] = comb['Sex'].map({'male':0,'female':1})

# drop Pclass, Embarked and Age features

comb = comb.drop(columns=['Pclass','Embarked','Age'])

comb.head()

df_train = comb.loc[comb['Survived'].isin([np.nan]) == False]

df_test  = comb.loc[comb['Survived'].isin([np.nan]) == True]

print(df_train.shape)

df_train.head()

from sklearn.model_selection import GridSearchCV

feature_names = ['Sex','P1','P2','P3','EQ','ES','EC','NoAge','Minor']

from sklearn.svm import SVC

model = SVC()

param_grid = {'C':[1,2,5,10,20,50]} 

grs = GridSearchCV(model, param_grid=param_grid, cv = 10, n_jobs=1, return_train_score = False)

grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))

print("BEST PARAMETERS " + str(grs.best_params_))

gpd = pd.DataFrame(grs.cv_results_)

print("ESTIMATED ACCURACY OF MODEL UNSEEN:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))

pred = grs.predict(np.array(df_test[feature_names]))

sub = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred})

sub.to_csv('submissionSVM.csv', index = False, float_format='%1d')

sub.head()

![SVM](https://user-images.githubusercontent.com/61597800/126331276-c077c669-bb40-404d-9af4-6a0817a5297c.PNG)


<h1>CODE BY 64238-ASAD & 64359-SHAAN</h1>

<h3>KNN CODE:</h3>


from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn import preprocessing

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

from sklearn.preprocessing import StandardScaler

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

import re

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

# File picking

%matplotlib inline

sns.set()

import os

df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

df_train.info()

df_train.describe()

sns.countplot(x='Survived',data=df_train)

#no of non survived persons are more

df_test['Survived']=0

df_test[['PassengerId','Survived']].to_csv('no_survivors.csv',index=False)#first model


sns.countplot(x='Sex',data=df_train)

sns.factorplot(x='Survived',col='Sex',kind='count',data=df_train)


df_train.groupby(['Sex']).Survived.sum()

print(df_train[df_train.Sex=='female'].Survived.sum()/df_train[df_train.Sex=='female'].Survived.count())

print(df_train[df_train.Sex=='male'].Survived.sum()/df_train[df_train.Sex=='male'].Survived.count())


df_test['Survived']=df_test.Sex=='female'

df_test['Survived']=df_test.Survived.apply(lambda x:int(x))

df_test[['PassengerId','Survived']].to_csv('women_survive.csv',index=False)#second model


survived_train=df_train.Survived

data=pd.concat([df_train.drop(['Survived'],axis=1),df_test.drop(['Survived'],axis=1)])

data.head()


data['Age']=data.Age.fillna(data.Age.mean())

data['Fare']=data.Fare.fillna(data.Fare.mean())


data['Embarked']=data.Embarked.fillna('S')

data['Embarked']=data.Embarked.map({'S':0,'C':1,'Q':2})

data['Sex']=data.Sex.map({'female':0,'male':1})


data['Has_cabin']=data.Cabin.apply(lambda x:0 if type(x)==float else 1)

data['Title']=data.Name.apply(lambda x:re.search('([A-Z][a-z]+)\.',x).group(1))

data['Title']=data.Title.replace({'Mlle':'Miss','Mme':'Mrs','Ms':'Miss','Master':'Mr'})

data['Title']=data.Title.replace(['Don','Dona','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer'],'Special')


data['Title']=data.Title.fillna(0)

data['Title']=data.Title.map({'Mr':0,'Mrs':1,'Miss':2,'Special':3})

data['CatAge']=pd.qcut(data.Age,q=4,labels=False)

data['CatFare']=pd.qcut(data.Fare,q=4,labels=False)

data['fam_size']=data.Parch+data.SibSp+1

data['IsAlone']=0

data.loc[data['fam_size']==1,'IsAlone']=1

data.drop(['Cabin','Name','PassengerId','Ticket','fam_size'],axis=1,inplace=True)

data=data.drop(['Age','Fare'],axis=1)

data=data.drop(['SibSp','Parch'],axis=1)

data.head(20)

data_dum=pd.get_dummies(data,drop_first=True)

data_train=data_dum.iloc[:891]

data_test=data_dum.iloc[891:]

X=data_train.values

test=data_test.values

y=survived_train.values

clf=KNeighborsClassifier()

k = np.arange(20)+1

parameters = {'n_neighbors': k}

clf_cv=GridSearchCV(clf,parameters,cv=10)

clf_cv.fit(X,y)

y_pred=clf_cv.predict(test)

df_test['Survived']=y_pred

df_test[['PassengerId','Survived']].to_csv("KNN_model.csv",index=False)

![Capture](https://user-images.githubusercontent.com/64367202/126348387-378d92cb-5222-40c8-baf7-67d63edceb99.PNG)


<h1>LINEAR CLASSIFIER CODE: </h1>

<h3>CODE BY 64359-SHAAN , 64238-ASAD & 63437-AHSAN IQBAL(107339)</h3>

import numpy as np

import pandas as pd


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")

train_data.describe()

test_data.describe()

import matplotlib.pyplot as plt

%matplotlib inline

plt.subplots(figsize=(7, 5))

plt.boxplot(train_data['Fare'])

plt.title('Boxplot of Fare')

plt.show()

# Retrieve rows with Fare greater than 500

train_data[train_data['Fare']>500]

# Retrieve rows with Fare equal to 0

train_data[train_data['Fare']==0]

# Number of missing values in each column in train data

train_data.isnull().sum()


# Number of missing values in each column in test data

test_data.isnull().sum()

# Function to extract title from passenger's name

def extract_title(df):
    title = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
    return title

# Count of each title in train data

train_data['Title'] = extract_title(train_data)

train_data['Title'].value_counts()

# Count of each title in test data

test_data['Title'] = extract_title(test_data)

test_data['Title'].value_counts()

# Function to map titles to main categories

def map_title(df):
    title_category = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
    }
    new_title = df['Title'].map(title_category)
    return new_title

# Count of each title in train data after mapping

train_data['Title'] = map_title(train_data)

train_data['Title'].value_counts()

# Count of each title in test data after mapping

test_data['Title'] = map_title(test_data)

test_data['Title'].value_counts()

# Group train data by 'Pclass', 'Title' and calculate the median age

train_data.groupby(['Pclass', 'Title']).median()['Age']

# Function to identify passengers who have the title 'Miss' and, 1 or 2 value in the 'Parch' column

def is_young(df):
    young = []
    for index, value in df['Parch'].items():
        if ((df.loc[index, 'Title'] == 'Miss') and (value == 1 or value == 2)):
            young.append(1)
        else:
            young.append(0)
    return young

# Group train data by 'Pclass', 'Title', 'Is_Young(Miss)' and calculate the median age

train_data['Is_Young(Miss)'] = is_young(train_data)

grouped_age = train_data.groupby(['Pclass', 'Title', 'Is_Young(Miss)']).median()['Age']

grouped_age

test_data['Is_Young(Miss)'] = is_young(test_data)

# Fill missing age values in train and test data

train_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)

train_data['Age'].fillna(grouped_age, inplace=True)

train_data.reset_index(drop=True, inplace=True)

test_data.set_index(['Pclass', 'Title', 'Is_Young(Miss)'], drop=False, inplace=True)

test_data['Age'].fillna(grouped_age, inplace=True)

test_data.reset_index(drop=True, inplace=True)

# Group train data by 'Pclass' and calculate the median fare

grouped_fare = train_data.groupby('Pclass').median()['Fare']

grouped_fare

# Fill the missing fare value in test data

test_data.set_index('Pclass', drop=False, inplace=True)

test_data['Fare'].fillna(grouped_fare, inplace=True)

test_data.reset_index(drop=True, inplace=True)

# Drop unnecessary rows and columns

train_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)

test_data.drop(columns=['Name', 'Cabin', 'Ticket', 'Is_Young(Miss)'], inplace=True)

train_data.dropna(subset=['Embarked'], inplace=True)

# Missing values in train data after data cleaning

train_data.isnull().sum()

# Missing values in test data after data cleaning

test_data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

# Encode 'Sex' variable values

le = LabelEncoder()

train_data['Sex'] = le.fit_transform(train_data['Sex'])

test_data['Sex'] = le.transform(test_data['Sex'])

# Convert 'Embarked' and 'Title' into dummy variables

train_data = pd.get_dummies(train_data, columns=['Embarked', 'Title'])

test_data = pd.get_dummies(test_data, columns=['Embarked', 'Title'])

# Pairwise correlation of columns

corr = train_data.corr()

corr

from sklearn.preprocessing import MinMaxScaler

# Apply feature scaling using MinMaxScaler

scaler = MinMaxScaler()

train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:])

test_data.iloc[:, 1:] = scaler.transform(test_data.iloc[:, 1:])

train_data.head()

X_train, X_test, y_train = train_data.iloc[:, 2:], test_data.iloc[:, 1:], train_data['Survived']

lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)

# Test score

y_preds = lda.predict(X_test)

submission=(y_preds)

# Function to generate submission file to get test score

def submission(preds):

test_data['Survived'] = preds
    predictions = test_data[['PassengerId', 'Survived']]
    predictions.to_csv('submissionLinear.csv', index=False)
    

![linear](https://user-images.githubusercontent.com/61597800/126363682-47bbc287-d4f8-47b6-bf85-8d35bd478957.PNG)



