#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 07:04:41 2021

@author: alexlange
"""

import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.insert(1, 'Users/alexlange/Documents/GitHub/6103_12T_21FA_IntroDM/mod4/Week11/')
import ds6103 as dc
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
#%%
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
#%%
data = pd.read_csv("/Users/alexlange/Downloads/train.csv")
test = pd.read_csv("/Users/alexlange/Downloads/test.csv")
data = data.drop(['id'],axis=1)
test = test.drop(['id'],axis=1)
#%%
data['Flight Distance'] = data['Flight Distance']/max(data['Flight Distance'])
#%%
data = data.drop(["Gender","Unnamed: 0","Age"],axis=1) # Omitting for simplicity because this is not in scope 
#%%
def cleanDfIncome(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (thisamt == "neutral or dissatisfied"): return float(0)
  if (thisamt == "satisfied"): return float(1)
data['satisfaction'] = data.apply(cleanDfIncome, colname='satisfaction', axis=1)
#%%
def clean_cust(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (thisamt == "Loyal Customer"): return float(1)
  if (thisamt == "disloyal Customer"): return float(0)
data['Customer Type'] = data.apply(clean_cust, colname='Customer Type', axis=1)
#%%
def clean_bus(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (thisamt == "Business travel"): return float(1)
  if (thisamt == "Personal Travel"): return float(0)
data['Type of Travel'] = data.apply(clean_bus, colname='Type of Travel', axis=1)
#%%
def clean_class(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (thisamt == "Eco"): return float(0)
  if (thisamt == "Eco Plus"): return float(1)
  if (thisamt == "Business"): return float(2)
data['Class'] = data.apply(clean_class, colname='Class', axis=1)
#%%
def dep_delay(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (np.isnan(thisamt) == True): return float(0)
  else: return thisamt
data['Departure Delay in Minutes'] = data.apply(dep_delay, colname='Departure Delay in Minutes', axis=1)
#%%
def dep_arr(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname]
  if (np.isnan(thisamt) == True): return float(0)
  else: return thisamt
data['Arrival Delay in Minutes'] = data.apply(dep_arr, colname='Arrival Delay in Minutes', axis=1)
#%%
#%%
ydata = data["satisfaction"]
data = data.drop("satisfaction",axis=1)
#%%
X_train, X_test, y_train, y_test = train_test_split(data,ydata,test_size=.2,train_size=.8 )
selector1 = SelectKBest(chi2, k=10)
#y_train=data["satisfaction"]
x_new = selector1.fit(X_train, y_train)

print('Top 10 Features are in order of importance: \n' + str(X_train.columns[selector1.get_support(indices=True)]) + '\n' + 'Fitting Coefficients are: ' + str(x_new.scores_) + '\n'+ 'and P-values for data are: ' + str(x_new.pvalues_))

#%% Some initial EDA for the plots
sns.histplot(x='Inflight wifi service',hue="satisfaction",data=data,multiple="stack").set_title("Satisfaction levels corresponding to Inflight wifi service rating")
plt.plot()
#%%
sns.histplot(x='Baggage handling',hue="satisfaction",data=data,multiple="stack").set_title("Satisfaction levels corresponding to Baggage handling")
plt.plot()

#%% 
# clf1 = LogisticRegression()
# clf2 = LinearSVC()
# clf3 = SVC(kernel="linear")
# clf4 = SVC()
# clf5 = DecisionTreeClassifier()
# clf6 = KNeighborsClassifier(n_neighbors=3) 
# classifiers = [clf1,clf2] # use even numbers to avoid issue for now
# # classifiers.append(clf3)
# classifiers.append(clf4)
# classifiers.append(clf5)
# classifiers.append(clf6)
## about 10 min
# Fit the classifiers
# for c in classifiers:
#     c.fit(xadmit,yadmit)

# # Plot the classifiers
# dc.plot_classifiers(xadmit.values, yadmit.values, classifiers)
#%%
# You can try adding clf3 and clf6, but KNN takes a long time to render.
#xadmit = data[['Inflight wifi service', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'Inflight service']]
xadmit = data[['Customer Type','Type of Travel','Class','Flight Distance','Inflight wifi service','Online boarding','Seat comfort','Inflight entertainment',
               'On-board service','Leg room service','Departure Delay in Minutes','Arrival Delay in Minutes']]
xadmit2 = data[['Customer Type','Type of Travel','Class','Flight Distance','Inflight wifi service','Online boarding','Seat comfort','Inflight entertainment',
               'On-board service','Leg room service']]
yadmit = ydata
#xadmit2 = data[['Inflight wifi service', 'Food and drink']]  # just two features so that we can plot 2-D easily
# yadmit = dfadmit['admit']
#%% about 10 min
# Fit the classifiers
# for c in classifiers:
#     c.fit(xadmit,yadmit)

# # Plot the classifiers
# dc.plot_classifiers(xadmit.values, yadmit.values, classifiers)
#%%
# for c in classifiers:
#     c.fit(xadmit2,yadmit)

# # Plot the classifiers
# dc.plot_classifiers(xadmit2.values, yadmit.values, classifiers)
#%%
# fig.legend(['N/D', 'S', 'Wifi', 'F&D'],
#                  loc='upper right', fancybox=True, scatterpoints=1)
#%% about 5 min
X_train, X_test, y_train, y_test = train_test_split(xadmit, yadmit)
svc = SVC(probability=True)
saved = svc.fit(X_train,y_train)
#%%
X_train2, X_test2, y_train2, y_test2 = train_test_split(xadmit2, yadmit)
svc2 = SVC(probability=True)
saved2 = svc.fit(X_train2,y_train2)
#%%
score = roc_auc_score(y_train, saved.predict_proba(X_train)[:, 1])
#%%
score2 = roc_auc_score(y_train2, saved2.predict_proba(X_train2)[:, 1])
#%%
lr_fpr, lr_tpr, _ = roc_curve(y_test, saved.predict_proba(X_test)[:, 1])
#%%
lr_fpr2, lr_tpr2, _ = roc_curve(y_test2, saved2.predict_proba(X_test2)[:, 1])

#%%
plt.plot(lr_fpr, lr_fpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("SVC ROC curve")
plt.legend(["No Skill AUC = .5","SVC AUC = .921"])
plt.plot()
#%%
plt.plot(lr_fpr2, lr_fpr2, linestyle='--', label='No Skill')
plt.plot(lr_fpr2, lr_tpr2, marker='.', label='7 Feature SVC')
plt.plot(lr_fpr, lr_tpr, marker='.', label='SVC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("SVC ROC curve")
plt.legend(["No Skill AUC = .5","7 Feature SVC AUC = .979","10 Feature SVC AUC = .921"])
plt.plot()
#%%
print(f'svc train score:  {svc.score(X_train,y_train)}')
print(f'svc test score:  {svc.score(X_test,y_test)}')
#%%
print(f'svc train score:  {svc.score(X_train2,y_train2)}')
print(f'svc test score:  {svc.score(X_test2,y_test2)}')
#%%
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))
#%%
print(confusion_matrix(y_test2, svc.predict(X_test2)))
print(classification_report(y_test2, svc.predict(X_test2)))
#%%
from sklearn.model_selection import cross_val_score
tree_cv_acc = cross_val_score(saved, X_train, y_train, cv= 10, scoring='accuracy',n_jobs = -1)
#%%
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
features_names = ['Customer Type','Type of Travel','Class','Flight Distance','Inflight wifi service','Online boarding',
                  'Seat comfort','Inflight entertainment','On-board service','Leg room service','Departure Delay in Minutes',
                  'Arrival Delay in Minutes']
svm = SVC(kernel='linear')
svm.fit(X_train,y_train)
f_importances(svm.coef_, features_names)
#%%
test = roc_auc_score(y_test,saved)
#test = roc_auc_score(y_test,svc.fit(X_train,y_train))
#%%
test = pd.DataFrame(y_test)
lr_auc = roc_auc_score(y_test, svc.predict())
a,b = roc_curve(test, svc.predict_proba(X_test))


