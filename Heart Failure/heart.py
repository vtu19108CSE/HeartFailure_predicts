import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

df=pd.read_csv("hf.csv")

X,y=df.iloc[:,:-1],df['DEATH_EVENT']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,shuffle=True)

#DEcisionTreeClassifier
from sklearn import tree
dt=tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)

prediction=dt.predict(X_test)

#randomforest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

import pickle
pickle.dump(rf,open('hf1.pkl','wb'))