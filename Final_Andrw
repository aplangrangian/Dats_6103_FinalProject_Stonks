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
x=airline.drop(['satisfaction',"Arrival Delay in Minutes"], axis=1)
y=airline["satisfaction"]
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=1)

# %%
###DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=1)
dtr.fit(x_train,y_train)

# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
y_predict=dtr.predict((x_test))
print(accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
# %%
###Feature Importance
feature_importances=pd.DataFrame({'features':x_train.columns,'feature_importance':dtr.feature_importances_})
feature_importances1=feature_importances.sort_values(by='feature_importance',ascending=False)
sns.barplot(feature_importances1["features"],feature_importances1["feature_importance"])
plt.xticks(rotation=90)
##From this graph, we can tell that online boarding, wifi service, and type of travel are the top 3 important features
##So the company could work more on the these 3 factors to improve satisfaction
# %%

# %%

# %%

# %%
