import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('Titanic_train.csv')

test = pd.read_csv('Titanic_test.csv')

train['Died'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',figsize=(6, 3),stacked=True)                                                      
plt.show()

figure = plt.figure(figsize=(10, 4))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
         stacked=True, bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

df=train.drop(['Name','Ticket','Cabin','PassengerId','Died'], axis=1)
df.head(10)

df.isnull().sum()

#Converting the categorical features 'Sex' and 'Embarked' into numerical values 0 & 1
df.Sex=df.Sex.map({'female':0, 'male':1})
df.Embarked=df.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df.head()

#Mean age of each sex
mean_age_men=df[df['Sex']==1]['Age'].mean()
mean_age_women=df[df['Sex']==0]['Age'].mean()

#Filling all the null values in 'Age' with respective mean age
df.loc[(df.Age.isnull()) & (df['Sex']==0),'Age']=mean_age_women
df.loc[(df.Age.isnull()) & (df['Sex']==1),'Age']=mean_age_men

df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

#Doing Feature Scaling to standardize the independent features present in the data in a fixed range
df.Age = (df.Age-min(df.Age))/(max(df.Age)-min(df.Age))
df.Fare = (df.Fare-min(df.Fare))/(max(df.Fare)-min(df.Fare))
df.describe()

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived'], axis=1), 
df.Survived, test_size= 0.2, random_state=0, stratify=df.Survived)

from sklearn.linear_model import LogisticRegression
lrmod = LogisticRegression()
lrmod.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
y_predict = lrmod.predict(X_test)
accuracy_score(y_test, y_predict)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
plt.figure(figsize = (7,4))
cma=confusion_matrix(y_test, y_predict)
sns.heatmap(cma,annot=True)
plt.show()

#Viewing test data
test.head()
df1=test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df1

#Converting the categorical features 'Sex' and 'Embarked' into numerical values 0 & 1
df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
df1.head()

df1.isnull().sum()
mean_age_men2=df1[df1['Sex']==1]['Age'].mean()
mean_age_women2=df1[df1['Sex']==0]['Age'].mean()

df1.loc[(df1.Age.isnull()) & (df1['Sex']==0),'Age']=mean_age_women2
df1.loc[(df1.Age.isnull()) & (df1['Sex']==1),'Age']=mean_age_men2
df1['Fare']=df1['Fare'].fillna(df1['Fare'].mean())

df1.Age = (df1.Age-min(df1.Age))/(max(df1.Age)-min(df1.Age))
df1.Fare = (df1.Fare-min(df1.Fare))/(max(df1.Fare)-min(df1.Fare))
df1.describe()

prediction = lrmod.predict(df1)
prediction

submission = pd.DataFrame({"PassengerId": test["PassengerId"],
                            "Survived": prediction})
submission.to_csv('submission.csv', index=False)
prediction_df = pd.read_csv('submission.csv')

sns.countplot(x='Survived', data=prediction_df)
plt.show()

# Predict class labels (0 or 1)
y_pred = model.predict(X_test)

# Predict probabilities for ROC-AUC
y_proba = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import f1_score, roc_auc_score

f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
}).sort_values(by='Coefficient', ascending=False)

print(coef_df)
