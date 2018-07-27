# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:43:47 2018

@author: chenho
"""

# public libarary
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# self-defined module
import sys
#sys.path.append(r'C:\Users\chenho\Documents\EY\EY Private\Self-defined-Functions') # Add a directory into sys path
sys.path.append(r'G:\Data Science\Self-defined Functions') # Add a directory into sys path

import SupportingFunctions as SF
import EDD as EDD
import woe_hc as WOE
import MachineLearningModels as MLM

#### data loading ####
dir_data = r"G:\Data Science\Kaggle Competition\Titanic Sinking\data"
dir_output = r"G:\Data Science\Kaggle Competition\Titanic Sinking\output"

df_train = SF.load_csv(dir_data + "\\train.csv")
df_test = SF.load_csv(dir_data + "\\test.csv")

df_edd_num, df_edd_cat = EDD.EDD(df_train, ls_force_categorical=['Survived', 'Pclass'])
df = df_train.copy()

#### data exploration ####
df['Age'].max()
sns.distplot(df['Age'].fillna(99))

sns.distplot(df['Fare'])

#### Feature Engineering ####
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df['Cabin_ct'] = df['Cabin'].apply(lambda x: 0 if str(x) == "nan" else (str(x).count(" ")+1))
df['Cabin_class'] = df['Cabin'].apply(lambda x: 'nan' if str(x) == "nan" else x[:1])

# Apply WOE transfromation to categorical variables
dict_woe_df = {}
dict_woe = {}

woe_cat = WOE.WoE(qnt_num=7, min_block_size=50, v_type='d', t_type='b')
for col in ['Sex', 'Embarked', 'Pclass', 'Title', 'Cabin_class']:
    woe_temp = woe_cat.categorical(x=df[col], y=df['Survived'])
    dict_woe[col] = woe_temp
    dict_woe_df[col] = woe_temp.df
    df[col] = woe_temp.df.woe
    woe_temp.plot()
    
df.columns.tolist() 
var_x = ['Pclass',
         'Sex',
         'Age',
         'SibSp',
         'Parch',
         'Fare',
         'Embarked',
         'Title',
         'Cabin_ct',
         'Cabin_class']
var_y = 'Survived'

#### Variable Selection ####
df_corr = SF.my_corr(df[var_x], 'pearson').sort_values(['corr'], ascending=[False])

def my_iv(dict_woe_df, col):
    df_iv = dict_woe_df[col].groupby('X').agg({'Y':['sum','count'], 'woe':'max'})
    df_iv.columns = ['bad', 'total', 'woe']
    df_iv['good'] = df_iv['total'] - df_iv['bad']
    df_iv['rate_diff'] = (df_iv['good'] / df_iv['good'].sum()) - (df_iv['bad'] / df_iv['bad'].sum())
    
    return (df_iv['rate_diff'] *df_iv['woe']).sum()

my_iv(dict_woe_df, 'Sex')
my_iv(dict_woe_df, 'Title')

var_x.remove('Sex')

#### Modeling ####
# Train & Test Split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(df[var_x], df[var_y], test_size=0.33, random_state=66)  

# RF
model_rf, df_importance_rf = MLM.my_sklearn_rf(X_train, y_train, param = {"n_estimators":500,
                                                                          "max_features":0.6,
                                                                          "max_depth":8,
                                                                          "min_samples_split":0.005,
                                                                          "min_samples_leaf":0.001,
                                                                          "random_state":66,
                                                                          "min_impurity_decrease":0.006})
SF.plot_roc(model_rf, X_train, X_test, y_train, y_test)

scores = cross_val_score(model_rf, df[var_x], df[var_y], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# XGB
model_xgb, df_importance_xgb = MLM.my_API_XGBoost(X_train, y_train, X_test, y_test, param={"learning_rate":0.8,
                                                                                           "n_estimators":100, 
                                                                                           "max_depth":10, 
                                                                                           "colsample_bytree":0.6,
                                                                                           "colsample_bylevel":0.1,
                                                                                           "subsample":0.8,
                                                                                           "gamma":7,
                                                                                           "random_state":66})
SF.plot_roc(model_xgb, X_train, X_test, y_train, y_test)

scores = cross_val_score(model_xgb, df[var_x], df[var_y], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Prediction
df = df_test.copy()

#### Feature Engineering ####
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

df['Cabin_ct'] = df['Cabin'].apply(lambda x: 0 if str(x) == "nan" else (str(x).count(" ")+1))
df['Cabin_class'] = df['Cabin'].apply(lambda x: 'nan' if str(x) == "nan" else x[:1])

for col in ['Sex', 'Embarked', 'Pclass', 'Title', 'Cabin_class']:
    woe_temp = dict_woe_df[col][['X','woe']].drop_duplicates()
    df = pd.merge(df, woe_temp, left_on=col, right_on='X', how='left')
    df = df.drop(columns=['X'])
    df = df.drop(columns=[col])
    df = df.rename(index=str, columns={"woe": col})

df_pred = model_rf.predict(df[var_x])
submission = pd.DataFrame({"PassengerId": df_test["PassengerId"],
                           "Survived": df_pred})
submission.to_csv(dir_output + "\\submission.csv", index=False)