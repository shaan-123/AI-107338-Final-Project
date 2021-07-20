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

<h2>KNN CODE:</h2>


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
