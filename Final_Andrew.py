#%%
import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import crosstab
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from seaborn.palettes import husl_palette 

#%%
airline= pd.read_csv("train.csv")
airline.head()
# %%
###drop unnamed
airline = airline.drop('Unnamed: 0', 1)
airline.head()
# %%
###drop id
airline = airline.drop('id', 1)
airline.head()
# %%
###Change Gender to numerical
Gender_dict={"Female":0,"Male":1}
airline["Gender"]=airline["Gender"].map(Gender_dict)
airline.head()
# %%
###Change Customer Type to numerical
Type_dict={"Loyal Customer":1,"disloyal Customer":0}
airline["Customer Type"]=airline["Customer Type"].map(Type_dict)
airline.head()
# %%
###Change Customer Type of Travel to numerical
Travel_dict={"Personal Travel":0,"Business travel":1,}
airline["Type of Travel"]=airline["Type of Travel"].map(Travel_dict)
airline.head()
# %%
###Change Customer Class to numerical
Class_dict={"Eco":0,"Eco Plus":1,"Business":2}
airline["Class"]=airline["Class"].map(Class_dict)
airline.head()
# %%
##Fill NA in Arrival Delay in Minutes and convert it to int
airline['Arrival Delay in Minutes'] = airline['Arrival Delay in Minutes'].fillna(0)
airline['Arrival Delay in Minutes'].astype("int")
# %%
###Change satisfaction to numerical
satisfaction_dict={"satisfied":1,"neutral or dissatisfied":0}
airline["satisfaction"]=airline["satisfaction"].map(satisfaction_dict)
airline.head()
# %%
##### Smart Question:
#####Is there a substantial impact in the satisfaction with class?
##H0=There is no relationship between satisfaction and class
##HA=There is a relationship between satisfaction and class
import matplotlib.ticker as ticker
a=sns.histplot(data=airline,binwidth=0.4, x="Class", hue="satisfaction", multiple="stack")
a.xaxis.set_major_locator(ticker.MultipleLocator(1))

## We can tell from the graph that class does have a impact on satisfaction,then we make a chi-square test
# %%
## Build the corss table
satisfaction=airline["satisfaction"]
Class=airline["Class"]
crosstable= pd.crosstab(satisfaction,Class)
crosstable
# %%
# %%
## find the p value
from scipy.stats import chi2_contingency
stat, p, dof, expected=chi2_contingency(crosstable)
p

### from the p value, we can tell that p is less thatn 0.05,so we can reject null hypothesis, and tell that there is a relationship between class and satisfaction,
### and class does impact satisfaction
# %%
####Feature Selection


# %%
###Split test and train
from sklearn.model_selection import train_test_split
x=airline[["Customer Type", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service"]]
y=airline["satisfaction"]
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=1)

# %%
###DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier( criterion='entropy',random_state=1)
dtc.fit(x_train,y_train)

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
y_train_predict=dtc.predict((x_train))
y_test_predict=dtc.predict((x_test))
# Evaluate train-set accuracy
print('train set evaluation: ')
print(accuracy_score(y_train, y_train_predict))
print(confusion_matrix(y_train, y_train_predict))
print(classification_report(y_train, y_train_predict))
# Evaluate test-set accuracy
print('test set evaluation: ')
print(accuracy_score(y_test, y_test_predict))
print(confusion_matrix(y_test, y_test_predict))
print(classification_report(y_test, y_test_predict))


# %%
###Feature Importance
feature_importances=pd.DataFrame({'features':x_train.columns,'feature_importance':dtc.feature_importances_})
feature_importances1=feature_importances.sort_values(by='feature_importance',ascending=False)
sns.barplot(feature_importances1["features"],feature_importances1["feature_importance"])
plt.xticks(rotation=90)
##From this graph, we can tell that online boarding, wifi service, and type of travel are the top 3 important features
##So the company could work more on the these 3 factors to improve satisfaction

# %%
maxlevel=None 
crit = 'gini' 
dtc1 = DecisionTreeClassifier(max_depth=2, criterion=crit, random_state=1)
dtc1.fit(x_train,y_train)
y_train_pred = dtc1.predict(x_train)
y_test_pred = dtc1.predict(x_test)
# Evaluate train-set accuracy
print('train set evaluation: ')
print(accuracy_score(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))
# Evaluate test-set accuracy
print('test set evaluation: ')
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))



# %%
####ROC AUC Score
from sklearn.metrics import roc_auc_score, roc_curve
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = dtc.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='DTC')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# %%



# %%
#####Tree plot
from sklearn import tree
a=["satisfied","not-satisfied"]
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc1, 
                   feature_names=x.columns,  
                   class_names=a,
                   filled=True)
# %%
