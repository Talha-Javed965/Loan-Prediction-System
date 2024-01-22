#!/usr/bin/env python
# coding: utf-8

# In[157]:


## LOADING PACKAGES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")


# In[90]:


# Reading Data
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")


# In[91]:


# Creating copies of Original Files

train_original = train.copy()
test_original = test.copy()


# In[92]:


# Understnding the structure of data we have and type of features it carries

train.columns


# In[93]:


test.columns


# In[94]:



print(train.head(5))


# In[95]:


train.dtypes


# In[96]:


# Checking the shape of dataset, number or rows and columns
train.shape


# In[97]:


test.shape


# In[98]:


# Now we will start the Univariate Analysis, where we look into the individual variables one by one  

train['Loan_Status'].value_counts()


# In[99]:


train['Loan_Status'].value_counts(normalize=True)


# In[100]:


train['Loan_Status'].value_counts().plot.bar()


# In[101]:


# Independent Variables (type of them is Categorical)

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Self_Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Credit_History')


# In[102]:


# Independent Variables (type of them is Ordinal)

plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(20,10), title = 'Property_Area')


# In[103]:


# Independent Variables (type of them is Numerical)
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize = (16,5))


# In[104]:


train.boxplot(column = 'ApplicantIncome', by = 'Education')
plt.suptitle(" ")


# In[105]:


# The above boxplot and subplot shows that is not normally distributed and there are lot of outliers in the dataset
Gender =pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))


# In[106]:


#From now onwards we will do Bivariate Analysis where each variable is compared with respect to Target variable(in this project Loan Status)

Married =pd.crosstab(train['Married'],train['Loan_Status'])
Dependents =pd.crosstab(train['Dependents'],train['Loan_Status'])
Education =pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed =pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))
Dependents.div(Dependents.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))
Education.div(Education.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))
Self_Employed.div(Self_Employed.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))


# In[107]:


bins = [0,1000,3000,42000]
group = ['low','average','high']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True, figsize = (4,4))


# In[108]:


train ['Total_Income'] = train ['ApplicantIncome'] + train ['CoapplicantIncome']
bins = [0,2500,4000,6000,81000]
group = ['low','average','high','very high']

train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin= pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar",stacked = True)
plt.xlabel("Total_Income")
P = plt.ylabel("Percentage")


# In[109]:


#Following is a heat map to understand the correlation between all numerical variables

train = train.drop(['Coapplicant_Income_bin','Total_Income_bin','Total_Income'], axis = 1)
train['Dependents'].replace('3+',3,inplace = True)
test['Dependents'].replace('3+',3,inplace = True)
train['Loan_Status'].replace('N',0,inplace = True)
train['Loan_Status'].replace('Y',1,inplace = True)

matrix = train.corr() 
f, ax = plt.subplots(figsize = (9,6))
sns.heatmap(matrix,vmax=.8,square= True, cmap = "BuPu");


# In[110]:


train.isnull().sum()


# In[111]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)


# In[113]:


train['Married'].fillna(train['Married'].mode()[0],inplace = True)


# In[114]:


train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)


# In[115]:


train['Loan_Amount_Term'].value_counts()


# In[116]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace = True)


# In[117]:


train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace = True)


# In[118]:


train.isnull().sum()


# In[119]:


test.isnull().sum()


# In[120]:


test['Gender'].fillna(train['Gender'].mode()[0],inplace = True)
test['Married'].fillna(train['Married'].mode()[0],inplace = True)
test['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace = True)
test['LoanAmount'].fillna(train['LoanAmount'].median(),inplace = True)


# In[121]:


test.isnull().sum()


# In[122]:


train['Loan_Amount_log'] = np.log(train['LoanAmount']) 
train['Loan_Amount_log'].hist(bins=20)
test['Loan_Amount_log'] = np.log(train['LoanAmount']) 


# In[123]:


train = train.drop('Loan_ID',axis =1)
test = test.drop('Loan_ID',axis =1)


# In[124]:


X = train.drop('Loan_Status',1)
Y = train.Loan_Status


# In[125]:


X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[126]:


from sklearn.model_selection import train_test_split


# In[128]:


x_train, x_cv, y_train, y_cv = train_test_split(X,Y,test_size = 0.3)


# In[129]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[132]:


model = LogisticRegression()
model.fit(x_train,y_train)
LogisticRegression(C = 1.0, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1,
                  max_iter = 100, multi_class = 'ovr', n_jobs = 1, penalty = '12', random_state = 1, solver = 'liblinear', tol = 0.0001
                    ,verbose = 0, warm_start = False )


# In[133]:


pred_cv = model.predict(x_cv)


# In[134]:


accuracy_score(y_cv,pred_cv)


# In[135]:


pred_test = model.predict(test)


# In[138]:


submission = pd.read_csv("sample_submission.csv")


# In[139]:


submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']


# In[140]:


submission['Loan_Status'].replace(0,'N',inplace = True )
submission['Loan_Status'].replace(1,'Y',inplace = True )


# In[141]:


pd.DataFrame(submission, columns = ['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[142]:


from sklearn.model_selection import StratifiedKFold


# In[153]:


i=1
kf = StratifiedKFold(n_splits=5, random_state = 1, shuffle= True) 
for train_index, test_index in kf.split(X,Y):
    print('\n{} of kfold {}' .format(i,kf.n_splits))
    xtr,xvl = X.iloc[train_index],X.iloc[test_index]
    ytr,yvl = Y.iloc[train_index],Y.iloc[test_index]
    model = LogisticRegression(random_state = 1)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl,pred_test)
    print('accuracy score',score)
    i+=1
    pred_test = model.predict(test)
    pred = model.predict_proba(xvl)[:,1]


# In[156]:


from sklearn import metrics
fpr,tpr, _ = metrics.roc_curve(yvl,pred)
auc = metrics.roc_auc_score(yvl,pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label = "validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[ ]:




