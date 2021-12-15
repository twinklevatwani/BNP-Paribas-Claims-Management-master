import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

os.chdir("F:\\Hackathon\\3. BNP Paribas Cardiff")

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# To view all the rows for any df summary options.
pd.options.display.max_rows = 150
# To view all the rows of the df.info()
pd.set_option('max_info_columns', 150)

## To view the individual column contents.
def rstr(df): return df.apply(lambda x: [x.unique()])

print(rstr(df))

df.isnull().sum().sort_values(ascending = False)
## Removing all columns, who have NA values more than 48619
col_names = list()
for i in range(len(df.columns)):
    if (df.iloc[:, i].isnull().sum() > 48618):
        col_names.append(df.columns[i])

df.drop(col_names, axis=1, inplace=True)

## Imputing the null values for the numerical colums(int and float)
for i in range(len(df.columns)):
    if (df.iloc[:, i].isnull().any() and df.iloc[:, i].dtypes in ['int64', 'float64']):
        df.iloc[:, i].fillna(df.iloc[:, i].mean(), inplace=True)

## Imputing the null values for the string columns.
for i in range(len(df.columns)):
    if (df.iloc[:, i].isnull().any() and df.iloc[:, i].dtypes == 'object'):
        df.iloc[:, i].fillna(df.iloc[:, i].mode()[0], inplace=True)

## View the string(object) columns.
df.select_dtypes(include = ['object']).columns

## Converting factor variables to categorical.(They get converted from object to int64)
char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
#label_mapping = {}
#for c in char_cols:
#    df[c],label_mapping[c] = pd.factorize(df[c])

temp_cat = df[char_cols]
temp_num = df[list(set(df)-(set(char_cols)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

df = pd.concat([temp_cat,temp_num],axis = 1)

r = list()
c = list()
cor = list()
for i in range(len(df.columns)):
    for j in range(len(df.columns)):
        if((df.iloc[:,i].corr(df.iloc[:,j]) > 0.70 or df.iloc[:,i].corr(df.iloc[:,j]) < -0.70) and df.iloc[:,i].corr(df.iloc[:,j])!=1):
            r.append(df.columns[i])
            c.append(df.columns[j])
            cor.append(df.iloc[:,i].corr(df.iloc[:,j]))

cor_df = pd.DataFrame({"1":r, "2":c, "corr value" : cor})

## removing the columns having correlations:v12,v21,v40, v114, v47, v79, v75, v91 (having the least correlations with the target variable as well)
## df.v91.corr(df.target)

#df.drop(['v12','v21','v40','v114','v47','v79','v75','v91'],axis = 1,inplace = True)
df.drop(['v12','v21','v40','v114','v47','v71'],axis = 1,inplace = True)
df_x = df.drop(['target','ID'], axis = 1)
df_y = df['target']

## Same Data Manipulation on the Test Data.

test.drop(col_names, axis=1, inplace=True)

## Imputing the null values for the numerical colums(int and float)
for i in range(len(test.columns)):
    if (test.iloc[:, i].isnull().any() and test.iloc[:, i].dtypes in ['int64', 'float64']):
        test.iloc[:, i].fillna(test.iloc[:, i].mean(), inplace=True)

## Imputing the null values for the string columns.
for i in range(len(test.columns)):
    if (test.iloc[:, i].isnull().any() and test.iloc[:, i].dtypes == 'object'):
        test.iloc[:, i].fillna(test.iloc[:, i].mode()[0], inplace=True)
final_id = test.ID
test.drop('ID', axis=1, inplace=True)

# char_cols = test.dtypes.pipe(lambda x: x[x == 'object']).index
# label_mapping = {}
# for c in char_cols:
#    test[c],label_mapping[c] = pd.factorize(test[c])

temp_cat = test[char_cols]
temp_num = test[list(set(test) - (set(char_cols)))]
temp_cat = temp_cat.apply(LabelEncoder().fit_transform)

test = pd.concat([temp_cat, temp_num], axis=1)

# test.drop(['v12','v21','v40','v114','v47','v79','v75','v91'],axis = 1,inplace = True)
test.drop(['v12', 'v21', 'v40', 'v114', 'v47', 'v71'], axis=1, inplace=True)

## Logistic Regression
model = LogisticRegression()
model.fit(df_x,df_y)
result = model.predict_proba(test)
accuracy = accuracy_score(df_y, model.predict(df_x))
rocauc= roc_auc_score(df_y, model.predict(df_x))

print(accuracy)
print(rocauc)

result_lr = pd.DataFrame({'ID': final_id,'PredictedProb': result[:,1]})
result_lr.to_csv('submission_lr.csv', index = False)

## RandomForest
modelrf = RandomForestClassifier()
modelrf.fit(df_x,df_y)
result = modelrf.predict_proba(test)
accuracy = accuracy_score(df_y, modelrf.predict(df_x))
rocauc= roc_auc_score(df_y, modelrf.predict(df_x))

print(accuracy)
print(rocauc)

result_rf = pd.DataFrame({'ID': final_id,'PredictedProb': result[:,1]})
result_rf.to_csv('submission_rf.csv', index = False)

## XGBoost
modelxg = XGBClassifier()
modelxg.fit(df_x,df_y)
result = modelxg.predict_proba(test)
accuracy = accuracy_score(df_y, modelxg.predict(df_x))
rocauc= roc_auc_score(df_y, modelxg.predict(df_x))

print(accuracy)
print(rocauc)

result_xg = pd.DataFrame({'ID': final_id,'PredictedProb': result[:,1]})
result_xg.to_csv('submission_xg.csv', index = False)