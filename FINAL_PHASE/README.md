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
