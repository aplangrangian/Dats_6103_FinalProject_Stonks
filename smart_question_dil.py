import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import the dataset as csv then read as pandas dataframe
curr_dir = os.getcwd()
test = curr_dir + "\\test.csv"
train = curr_dir + "\\train.csv"

df_test = pd.read_csv(test, index_col = "Unnamed: 0")
df_train = pd.read_csv(train, index_col = "Unnamed: 0")
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)

# check the dataframe
df_test.shape
df_train.shape
df_train.isnull().sum()  # 310 null values for Arrival Delay in Minutes
test = df_train.describe()  # median value for Arrival Delay is 0 and very highly skewed towards 0, so we replace with 0
df_train['Arrival Delay in Minutes'].fillna(0, inplace = True)
df_train.isnull().sum()
df_test['Arrival Delay in Minutes'].fillna(0, inplace = True)
df_test.isnull().sum()
# final check
sns.set_palette("colorblind")
sns.distplot(df_train['Arrival Delay in Minutes'])
plt.title("Distribution of Arrival Delay in Minutes")
plt.show()

df_train.describe()
df_train.info()
print(df_train.dtypes)
# let's put the categorical data in the columns variable for later
columns =['Gender','Customer Type', 'Type of Travel', 'Class','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness']
df_train_num=df_train.drop(labels=columns, axis=1)
df_test_num=df_test.drop(labels=columns, axis =1)

# looks good, now we can look at our data.
# first question is which of the categorical and which of the discrete influence satisfaction the most.

# gender vs satisfaction
sns.countplot(x='satisfaction', hue='Gender', data=df_train)
plt.title('Count plot for Gender')
plt.show()  # about a 50/50 split

# customertype vs satisfaction
sns.countplot(x='satisfaction', hue="Customer Type", data=df_train)
plt.title('Count plot for Customer Type')
plt.show()  # seems like loyal customers more likely to give feedback than non-loyal but were more often dissatisfied

# type of travel vs satisfaction
sns.countplot(x='satisfaction', hue="Type of Travel", data=df_train)
plt.title('Count plot for Type of Travel')
plt.show()  # those traveling for business are more likely to be satisfied

# Class vs satisfaction
sns.countplot(x='satisfaction', hue="Class", data=df_train)
plt.title('Count plot for Class')
plt.show()  # those traveling Eco are more likely not satisfied. Business class also more likely to be satisfied


# inflight wifi vs satisfaction
sns.countplot(x='satisfaction', hue="Inflight wifi service", data=df_train)
plt.title('Count plot for Inflight wifi service')
plt.show()  # multivariate might prefer a stacked graph. Looks like higher wifi service the low counts of dissatisfaction

# departure/arrival time convenient vs satisfaction
sns.countplot(x='satisfaction', hue="Departure/Arrival time convenient", data=df_train)
plt.title('Count plot for Departure/Arrival time convenient')
plt.show()  # same as above.


# Let's start prep by making Dummy variables for the variables
# Drop first so to avoid co-linearity
#df_train_dum=pd.get_dummies(df_train, columns=columns, drop_first=True)
#df_test_dum=pd.get_dummies(df_test, columns=columns, drop_first=True)



from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2

#dummy variable creation
lencoders = {}
for col in df_train.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    df_train[col] = lencoders[col].fit_transform(df_train[col])

lencoders_t = {}
for col in df_test.select_dtypes(include=['object']).columns:
    lencoders_t[col] = LabelEncoder()
    df_test[col] = lencoders[col].fit_transform(df_test[col])


# function for making dummy of 'satisfaction'
def transform_satisfaction(x):
    if x == 'satisfied':
        return 1
    elif x == 'neutral or dissatisfied':
        return 0
    else:
        return -1

# Apply function
df_train_num['satisfaction'] = df_train_num['satisfaction'].apply(transform_satisfaction)
df_test_num['satisfaction'] = df_test_num['satisfaction'].apply(transform_satisfaction)

# Chi2, top 10 features
# Let's use chi2 test of importance to find the top 10 categorical features
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(df_train)
modified_data = pd.DataFrame(r_scaler.transform(df_train), columns=df_train.columns)
print(modified_data.head)

selector = SelectKBest(chi2, k=10)
x_train=modified_data.drop(labels="satisfaction",axis=1) # df with only satisfaction and df with every other variable
y_train=modified_data["satisfaction"]
selector.fit(x_train, y_train)
x_new = selector.transform(x_train)
print(x_train.columns[selector.get_support(indices=True)])

# 7 features here are important ones for the selection process.
# We'll create a list of these and put them in our model.
important=["Customer Type", "Type of Travel", "Class", "Flight Distance", "Inflight wifi service", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service"]
df_train_important = df_train[important]
df_test_important = df_test[important]

from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve


# Create the training and test datasets.
#df_train_len=len(df_train_important)
#test=df_test_important[df_train_important]
#train=df_train[:df_train_len]
#x_train=train.drop(labels="satisfaction",axis=1) # df with only satisfaction and df with every other variable
#y_train=train["satisfaction"]

x_train_model,x_test_model,y_train_model,y_test_model=train_test_split(df_train_important,y_train,test_size=0.33,random_state=42)
print("x_train_model",len(x_train_model))
print("x_test_model",len(x_test_model))
print("y_train_model",len(y_train_model))
print("y_test_model",len(y_test_model))
print("test",len(test))



# Initiate the Logistic Regression Model and print the accuracy
logreg=LogisticRegression()
logreg.fit(x_train_model,y_train_model)

importance=logreg.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title('Bar plot of feature importance')
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.show()

acc_log_train=round(logreg.score(x_train_model,y_train_model)*100,2)
acc_log_test=round(logreg.score(x_test_model,y_test_model)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Test Accuracy: % {}".format(acc_log_test))
# Print the coef's
print(logreg.coef_)
# ROC predictions
ns_probs = [0 for _ in range(len(y_test_model))]
# predict probabilities
lr_probs = logreg.predict_proba(x_test_model)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test_model, ns_probs)
lr_auc = roc_auc_score(y_test_model, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test_model, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test_model, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill: ROC AUC=%.3f'%(ns_auc))
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic: ROC AUC=%.3f' % (lr_auc))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# Summary page for the model
y_predict=logreg.predict((x_test_model))
print(confusion_matrix(y_test_model, y_predict))
import statsmodels.api as sm
from sklearn.metrics import classification_report
logit_model=sm.Logit(y_train_model,x_train_model)
result=logit_model.fit()
print(result.summary())

# All features
df_train=df_train.drop(labels="satisfaction",axis=1)
x_train_model2,x_test_model2,y_train_model2,y_test_model2=train_test_split(df_train,y_train,test_size=0.33,random_state=42)
print("x_train_model2",len(x_train_model2))
print("x_test_model2",len(x_test_model2))
print("y_train_model2",len(y_train_model2))
print("y_test_model2",len(y_test_model2))
print("test",len(test))

# Initiate the Logistic Regression Model and print the accuracy
logreg2=LogisticRegression()
logreg2.fit(x_train_model2,y_train_model2)

importance2=logreg2.coef_[0]
for i,v in enumerate(importance2):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance2))], importance2)
plt.title('Bar plot of feature importance')
plt.xlabel("Features")
plt.ylabel("Coefficients")
plt.show()

acc_log_train=round(logreg2.score(x_train_model2,y_train_model2)*100,2)
acc_log_test=round(logreg2.score(x_test_model2,y_test_model2)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Test Accuracy: % {}".format(acc_log_test))
# Print the coef's
print(logreg2.coef_)

# ROC predictions
ns_probs2 = [0 for _ in range(len(y_test_model2))]
# predict probabilities
lr_probs2 = logreg2.predict_proba(x_test_model2)
# keep probabilities for the positive outcome only
lr_probs2 = lr_probs2[:, 1]
# calculate scores
ns_auc2 = roc_auc_score(y_test_model2, ns_probs2)
lr_auc2 = roc_auc_score(y_test_model2, lr_probs2)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc2))
print('Logistic: ROC AUC=%.3f' % (lr_auc2))
# calculate roc curves
ns_fpr2, ns_tpr2, _ = roc_curve(y_test_model2, ns_probs2)
lr_fpr2, lr_tpr2, _ = roc_curve(y_test_model2, lr_probs2)
# plot the roc curve for the model
plt.plot(ns_fpr2, ns_tpr2, linestyle='--', label='No Skill: ROC AUC=%.3f'%(ns_auc2))
plt.plot(lr_fpr2, lr_tpr2, marker='.', label='Logistic: ROC AUC=%.3f' % (lr_auc2))
# axis labels
plt.title("ROC Curve for All Features(Logistic)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# Summary page for the model
y_predict2=logreg2.predict((x_test_model2))
print(confusion_matrix(y_test_model2, y_predict2))
import statsmodels.api as sm
from sklearn.metrics import classification_report
logit_model2=sm.Logit(y_train_model2,x_train_model2)
result2=logit_model2.fit()
print(result2.summary())

