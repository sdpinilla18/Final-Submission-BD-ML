#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import pyreadr as pyr
import sklearn as sk
import matplotlib.pyplot as plt
import scipy as sc
import os
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import chi2
from sklearn.metrics import make_scorer
from numpy.random import normal
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from random import choices
from sklearn.model_selection import train_test_split
from mlens.ensemble import SuperLearner
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from scikeras.wrappers import KerasClassifier
from keras.utils import np_utils


# In[2]:


#Set directory:
os.chdir("C:/Users/hp/OneDrive - Universidad de los Andes/Documentos/Docs/Universidad/2022-2/Big Data/Trabajo Final")


# In[3]:


df=pd.read_csv("Base_2016_Clean.csv",low_memory=False)


# In[4]:


##Imputar missing values
imputer=KNNImputer(n_neighbors=19,weights='distance')
df_imp=imputer.fit_transform(df)
X=pd.DataFrame(df_imp,columns=df.columns)
Y=X["voto_ofrecido"]
X=X[[i for i in X.columns if i!="llaveper_n16" and i!="voto_ofrecido"]]


# In[5]:


## Train y test ####
X_train,X_test, y_train, y_test = train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=911)
##SMOTE para la train
oversample = SMOTE(sampling_strategy=0.8,k_neighbors=19,random_state=911,n_jobs=-1)
X_trains, y_trains = oversample.fit_resample(X_train, y_train)


# In[ ]:


#Redondeo
lista=['t_cuartos_hogar','t_personas', 'min_cabec','n_televisores', 'n_computadores', 'edad',
'valor_arriendo_pagado', 'hor_salud', 'n_duchas', 'n_otros_bienes']
for i in X_trains.columns:
    if i not in lista:
        X_trains[i]=X_trains[i].round(0)
for i in X_train.columns:
    if i not in lista:
        X_train[i]=X_train[i].round(0)
for i in X_test.columns:
    if i not in lista:
        X_test[i]=X_test[i].round(0)


# In[ ]:


ds=(df[['voto_ofrecido','edad','sexo','zona', 'nivel_educ_1','nivel_educ_2','nivel_educ_3','nivel_educ_4','sp_estrato_2',
'sp_estrato_3', 'sp_estrato_4', 'sp_estrato_5','sp_estrato_6','familias_accion', 'tenencia_vivienda_1',
 'tenencia_vivienda_2', 'tenencia_vivienda_3', 'tenencia_vivienda_4',
 'tenencia_vivienda_5','org_sindicato', 
'voto_alcaldia', 'evitar_iva']].describe(include="all"))
ds=ds.T
ds=ds[["count", "mean", "std", "50%"]]
ds=ds.round(4)
print(ds.to_latex())


# In[11]:


##Metrica
def wmir(y_true,y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # False negative rate
    FNR = FN/(TP+FN)
    #False positive rate
    FPR = FP/(FP+TN)
    penalty=FNR*(2/3)+(1/3)*FPR
    return penalty

wmirs=make_scorer(wmir, greater_is_better=False)

def FF(y_true,y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # False negative rate
    FNR = FN/(TP+FN)
    FPR = FP/(FP+TN)
    return FNR,FPR


# In[ ]:


# Modelos


# In[41]:


knn=KNeighborsClassifier()
neighbors=[11,13,15,17,19,21,23,25,27,29,31]
params = {
    'n_neighbors': neighbors,
    'weights': ['uniform','distance'],
}
knn_cv=GridSearchCV(knn,params,scoring=wmirs,n_jobs=-1, cv=10)
re_knncv=knn_cv.fit(X_trains,y_trains.values.ravel())


# In[42]:


re_knncv.best_params_


# In[43]:


print(wmir(y_test, re_knncv.predict(X_test)),FF(y_test, re_knncv.predict(X_test)))
print(wmir(y_trains, re_knncv.predict(X_trains)),FF(y_trains, re_knncv.predict(X_trains))) 


# In[44]:


parameters={'solver':['newton-cg', 'lbfgs', 'sag', 'saga'],'warm_start': [True,False]}
logistica=LogisticRegression(fit_intercept=True, penalty="none", random_state=911, max_iter=1000,n_jobs=-1)
logcv=GridSearchCV(logistica, parameters, scoring=wmirs)
resultslogistic_b=logcv.fit(X_trains, y_trains.values.ravel())

resultslogistic_b.best_params_


# In[56]:


print(wmir(y_test.values.ravel(),resultslogistic_b.predict(X_test)),FF(y_test.values.ravel(),resultslogistic_b.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultslogistic_b.predict(X_trains)),FF(y_trains.values.ravel(),resultslogistic_b.predict(X_trains)))


# In[46]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

grilla=np.linspace(0,1,100)
grilla=grilla.tolist()
reg_param={"reg_param": grilla}
qda=QuadraticDiscriminantAnalysis()
qdacv=GridSearchCV(qda, reg_param, scoring=wmirs,n_jobs=-1, cv=10)
modelqda=qdacv.fit(X_trains, y_trains.values.ravel())

modelqda.best_params_


# In[47]:


print(wmir(y_test.values.ravel(),modelqda.predict(X_test)),FF(y_test.values.ravel(),modelqda.predict(X_test)))
print(wmir(y_trains.values.ravel(),modelqda.predict(X_trains)),FF(y_trains.values.ravel(),modelqda.predict(X_trains)))


# In[335]:


qda2=QuadraticDiscriminantAnalysis(reg_param=0.425)
modelqda2=qda2.fit(X_trains, y_trains.values.ravel())


# In[336]:


print(wmir(y_test.values.ravel(),modelqda2.predict(X_test)),FF(y_test.values.ravel(),modelqda2.predict(X_test)))
print(wmir(y_trains.values.ravel(),modelqda2.predict(X_trains)),FF(y_trains.values.ravel(),modelqda2.predict(X_trains)))


# In[238]:


lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.525,n_components=1)
modellda=lda.fit(X_trains, y_trains.values.ravel())


# In[239]:


print(wmir(y_test.values.ravel(),modellda.predict(X_test)),FF(y_test.values.ravel(),modellda.predict(X_test)))
print(wmir(y_trains.values.ravel(),modellda.predict(X_trains)),FF(y_trains.values.ravel(),modellda.predict(X_trains)))


# In[54]:


l1 = np.linspace(0,1,10)
C = np.linspace(0,1,10)
parameters={'solver':['saga'],
            'warm_start': [True,False],
            'l1_ratio': l1,
            'C': C
            }
elastic=LogisticRegression(fit_intercept=True, penalty="elasticnet", random_state=911, max_iter=1000,n_jobs=-1)
encv=GridSearchCV(elastic, parameters, scoring=wmirs)
resultsen=encv.fit(X_trains, y_trains.values.ravel())

resultsen.best_params_


# In[55]:


print(wmir(y_test.values.ravel(),resultsen.predict(X_test)),FF(y_test.values.ravel(),resultsen.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultsen.predict(X_trains)),FF(y_trains.values.ravel(),resultsen.predict(X_trains)))


# In[303]:


elasticnw=LogisticRegression(fit_intercept=True,l1_ratio=1,C=0.025, penalty="elasticnet",solver='saga', random_state=911, max_iter=1000,n_jobs=-1)
modelenw=elasticnw.fit(X_trains, y_trains.values.ravel())


# In[304]:


print(wmir(y_test.values.ravel(),modelenw.predict(X_test)),FF(y_test.values.ravel(),modelenw.predict(X_test)))
print(wmir(y_trains.values.ravel(),modelenw.predict(X_trains)),FF(y_trains.values.ravel(),modelenw.predict(X_trains)))


# In[209]:


from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(n_estimators=50,max_depth=2,criterion='entropy',class_weight={0: 1,1: 3.6},max_features=None,bootstrap=True,n_jobs=-1,random_state=911)
resultsRF=RF.fit(X_trains, y_trains.values.ravel())


# In[210]:


print(wmir(y_test.values.ravel(),resultsRF.predict(X_test)),FF(y_test.values.ravel(),resultsRF.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultsRF.predict(X_trains)),FF(y_trains.values.ravel(),resultsRF.predict(X_trains)))


# In[517]:


import xgboost as xgb

XGB=xgb.XGBRFClassifier(n_estimators=50,max_depth=2,scale_pos_weight=3.55,objective='binary:logistic',booster='gbtree',grow_policy='lossguide',n_jobs=-1,random_state=911)
resultsXGB=XGB.fit(X_trains, y_trains.values.ravel())


# In[518]:


print(wmir(y_test.values.ravel(),resultsXGB.predict(X_test)),FF(y_test.values.ravel(),resultsXGB.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultsXGB.predict(X_trains)),FF(y_trains.values.ravel(),resultsXGB.predict(X_trains)))


# In[613]:


XGB2=xgb.XGBClassifier(n_estimators=35,max_depth=3,scale_pos_weight=16.75,reg_alpha=0,reg_lambda=1,objective='binary:logistic',booster='gbtree',grow_policy='depthwise',n_jobs=-1,random_state=911)
resultsXGB2=XGB2.fit(X_trains, y_trains.values.ravel())


# In[614]:


print(wmir(y_test.values.ravel(),resultsXGB2.predict(X_test)),FF(y_test.values.ravel(),resultsXGB2.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultsXGB2.predict(X_trains)),FF(y_trains.values.ravel(),resultsXGB2.predict(X_trains)))


# In[725]:


ANN=MLPClassifier(activation='relu',solver='adam',alpha=0.58,max_iter=50,verbose=True,hidden_layer_sizes=(2,4,8,16,32,64),random_state=911)
resultsANN=ANN.fit(X_trains, y_trains.values.ravel())


# In[726]:


print(wmir(y_test.values.ravel(),resultsANN.predict(X_test)),FF(y_test.values.ravel(),resultsANN.predict(X_test)))
print(wmir(y_trains.values.ravel(),resultsANN.predict(X_trains)),FF(y_trains.values.ravel(),resultsANN.predict(X_trains)))


# In[ ]:


import lightgbm as lgb
from lightgbm import LGBMClassifier
param = {'num_leaves': 31, 'objective': 'binary'}
train_data = lgb.Dataset(X_trains, label=y_trains)
param['metric'] = 'auc'
lgbclas=LGBMClassifier(n_estimators=106,scale_pos_weight=19,boosting_type='gbdt',reg_lambda=8 ,reg_alpha=9,num_leaves=9,max_depth=5,random_state=911,n_jobs=-1)
re_lgb=lgbclas.fit(X_trains,y_trains,eval_metric=wmir)
print(wmir(y_trains,re_lgb.predict(X_trains)))
print(FF(y_trains,re_lgb.predict(X_trains)))
print(wmir(y_test,re_lgb.predict(X_test)))
print(FF(y_test,re_lgb.predict(X_test)))
print(accuracy_score(y_test,re_lgb.predict(X_test)))


# In[ ]:


neurons=[(2,4,8,16,32,64),(1,2,4,8,16,32)]
activation=[('sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid'),('sigmoid','tanh','sigmoid','tanh','sigmoid','sigmoid'),
    ('sigmoid','relu','sigmoid','relu','sigmoid','sigmoid'),
    ('tanh','tanh','relu','tanh','tanh','tanh'),
    ('sigmoid','sigmoid','softmax','sigmoid','sigmoid','sigmoid')]
dropout=[(0.5,0.5,0.5,0.5,0.5),(0.1,0.1,0.1,0.1,0.1)]
batch_size=[128]

param_grid=dict(model_activation=activation,modelneurons=neurons,model_dropout=dropout, batch_size=batch_size)

def model(neurons=(1,2,4,8,16,32), activation=('sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','sigmoid'), dropout=(0.1,0.1,0.1,0.1,0.1)):
    model = Sequential()
    model.add(Dense(neurons[0], input_shape=(137,), activation=activation[0]))
    model.add(Dense(neurons[1], activation=activation[1]))
    model.add(Dropout(dropout[0]))
    model.add(Dense(neurons[2], activation=activation[2]))
    model.add(Dropout(dropout[1]))
    model.add(Dense(neurons[3], activation=activation[3]))
    model.add(Dropout(dropout[2]))
    model.add(Dense(neurons[4], activation=activation[4]))
    model.add(Dropout(dropout[3]))
    model.add(Dense(neurons[5], activation=activation[5]))
    model.add(Dropout(dropout[4]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='FalseNegatives')
    return model
modelo=KerasClassifier(model=model,epochs=80,verbose=1)
grid=GridSearchCV(estimator=modelo, param_grid=param_grid, scoring=wmir, n_jobs=-1, cv=5, verbose=1)
re_ann=grid.fit(X_trains,y_trains, class_weight={0: 1, 1: 2.8})
print(re_ann.best_params_)
y_pred=re_ann.predict(X_trains)
y_pred2=re_ann.predict(X_test)

