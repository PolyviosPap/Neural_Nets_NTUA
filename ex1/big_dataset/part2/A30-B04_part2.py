#!/usr/bin/env python
# coding: utf-8

# ### MLP με pipeline.
# 
# Στο τελικό στάδιο βελτιστοποίησης του MLP, θα προεπεξεργαστούμε τα δεδομένα μας, κάνοντας χρήση του pipeline.

# In[1]:


import pandas as pd
df = pd.read_csv("spambase.data", header=None)


# In[2]:


features_df = df.iloc[:,:-1]
labels_df = df.iloc[:,-1]

features = features_df.values
labels = labels_df.values


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[4]:


import numpy as np

np.bincount(y_test)
print(X_train.shape)


# In[5]:


from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# In[6]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[7]:


X_train_small = X_train[:500]
y_train_small = y_train[:500]


# In[8]:


np.arange(5, 200, 20)


# In[9]:


selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()
pca = PCA()
mlp = MLPClassifier()


# In[10]:


from sklearn.model_selection import GridSearchCV

n_components = [10, 20, 30, 40, 50, 55]


# In[11]:


pipe_mlp = Pipeline(steps=[('scaler', scaler), ('sampler', ros), ('pca', pca), ('mlp', mlp)], memory = 'tmp')


# In[31]:


# testing adam.
params = dict(
    pca__n_components = n_components,
    mlp__solver = ['adam'],
    mlp__activation = ['tanh'],
    mlp__max_iter = [800], 
    mlp__alpha = 10.0 ** -np.arange(7, 8), 
    mlp__hidden_layer_sizes = [30, 35],
    mlp__learning_rate = ['invscaling']
)

mlp_macro = GridSearchCV(pipe_mlp, params, cv=5, scoring='f1_micro', n_jobs=-1)
mlp_macro.fit(X_train_small, y_train_small)

mlp_macro.best_params_
mlp_macro.best_estimator_


# In[32]:


preds = mlp_macro.predict(X_test)
print(classification_report(y_test, preds))


# In[16]:


# testing lbfgs.
params = dict(
    pca__n_components = n_components,
    mlp__solver = ['lbfgs'],
    mlp__activation = ['logistic'],
    mlp__max_iter = [700, 800, 900], 
    mlp__alpha = 10.0 ** -np.arange(7, 10), 
    mlp__hidden_layer_sizes = [20, 25],
    mlp__learning_rate = ['constant', 'invscaling', 'adaptive']
)
                       
mlp_macro_v1 = GridSearchCV(pipe_mlp, params, cv=5, scoring='f1_macro', n_jobs=-1)
mlp_macro_v1.fit(X_train_small, y_train_small)

mlp_macro_v1.best_params_
mlp_macro_v1.best_estimator_


# In[17]:


preds_v1 = mlp_macro_v1.predict(X_test)
print(classification_report(y_test, preds_v1))


# Παρατηρούμε όμως πως όταν προεπεξεργαζόμαστε τα δεδομένα μας πριν το gridsearch τα scores είναι αρκετά χαμηλότερα από πριν, καθώς κυμαίνονται στο ~0.87.
# 
# Αυτό συμβαίνει επειδή η φύση των δεδομένων μας δε βοηθά τη διαδικασία του pipeline, αφού για παράδειγμα οι τελευταίες 3 στήλες περιέχουν τιμές μεγαλύτερες του 1 και οι υπόλοιπες είναι φραγμένες στο [0.0 , 1.0]

# In[21]:


parameters = {
    'solver': ['adam'], 
    'activation': ['logistic'],
    'max_iter': [800], 
    'alpha': 10.0 ** -np.arange(7, 8), 
    'hidden_layer_sizes':[25]
}

clf_h1 = MLPClassifier()
clf_grid_h1 = GridSearchCV(clf_h1, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_h1.fit(X_train, y_train)
preds_h1 = clf_grid_h1.predict(X_test)

clf_grid_h1.best_params_


# In[22]:


print(classification_report(y_test, preds_h1))


# In[ ]:




