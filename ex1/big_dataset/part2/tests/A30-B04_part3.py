#!/usr/bin/env python
# coding: utf-8

# ### MLP.
# Θα προσπαθήσουμε να βελτιστοποιήσουμε τον MLP, αρχικά χωρίς τη χρήση του pipeline, για να μειώσουμε τον όγκο των δεδομένων μας και να προχωρήσουμε έπειτα έχοντας μια καλύτερη εικόνα για τον MLP στο dataset μας.
# 
# Παραθέτουμε τα βήματα που ακολουθήσαμε, κάνοντας διαδοχικά gridsearch στον MLP δοκιμάζοντες μικρό πεδίο για κάθε μια υπερπαράμετρο κάθε φορά, κρατόντας επίσης τον solver σταθερό.

# In[26]:


import warnings 
warnings.filterwarnings('ignore')


# In[27]:


import pandas as pd
df = pd.read_csv("spambase.data", header=None)


# In[28]:


features_df = df.iloc[:,:-1]
labels_df = df.iloc[:,-1]

features = features_df.values
labels = labels_df.values


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# In[30]:


import numpy as np

np.bincount(y_test)
print(X_train.shape)


# In[31]:


from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# In[32]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[33]:


X_train_small = X_train[:500]
y_train_small = y_train[:500]


# In[34]:


np.arange(5, 200, 20)


# ### Solver: lbfgs Hidden (5, 20, 5)

# In[69]:


parameters = {
    'solver': ['lbfgs'], 
    'max_iter': [100, 250, 500], 
    'alpha': 10.0 ** -np.arange(1, 10), 
    'hidden_layer_sizes':np.arange(5, 20, 5)
}

clf = MLPClassifier()
clf_grid = GridSearchCV(clf, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid.fit(X_train, y_train)
preds = clf_grid.predict(X_test)

clf_grid.best_params_


# In[70]:


print(classification_report(y_test, preds))


# In[71]:


clf_grid.cv_results_


# ### Solver: sgd Hidden: (5, 20, 5)

# In[73]:


parameters = {
    'solver': ['sgd'], 
    'max_iter': [100, 250, 500], 
    'alpha': 10.0 ** -np.arange(1, 10), 
    'hidden_layer_sizes':np.arange(5, 20, 5)
}

clf_sgd_5 = MLPClassifier()
clf_grid_sgd_5 = GridSearchCV(clf_sgd_5, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_sgd_5.fit(X_train, y_train)
preds_sgd_5 = clf_grid_sgd_5.predict(X_test)

clf_grid_sgd_5.best_params_


# In[75]:


print(classification_report(y_test, preds_sgd_5))


# In[74]:


clf_grid_sgd_5.cv_results_


# ### Solver: adam Hidden: (5, 20, 5)

# In[76]:


parameters = {
    'solver': ['adam'], 
    'max_iter': [100, 250, 500], 
    'alpha': 10.0 ** -np.arange(1, 10), 
    'hidden_layer_sizes':np.arange(5, 20, 5)
}

clf_adam_5 = MLPClassifier()
clf_grid_adam_5 = GridSearchCV(clf_adam_5, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_adam_5.fit(X_train, y_train)
preds_adam_5 = clf_grid_adam_5.predict(X_test)

clf_grid_adam_5.best_params_


# In[79]:


print(classification_report(y_test, preds_adam_5))


# In[80]:


clf_grid_adam_5.cv_results_


# ### Activation (adam)

# In[81]:


parameters = {
    'solver': ['adam'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [200, 250, 350], 
    'alpha': [0.1, 0.2, 0.5], 
    'hidden_layer_sizes':[15, 20, 25]
}

clf_adam_a = MLPClassifier()
clf_grid_adam_a = GridSearchCV(clf_adam_a, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_adam_a.fit(X_train, y_train)
preds_adam_a = clf_grid_adam_a.predict(X_test)

clf_grid_adam_a.best_params_


# In[82]:


print(classification_report(y_test, preds_adam_a))


# In[83]:


clf_grid_adam_a.cv_results_


# ### Activation (adam) 2

# In[85]:


parameters = {
    'solver': ['adam'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [200, 250, 350], 
    'alpha': [0.1, 0.2, 0.5], 
    'hidden_layer_sizes':[20, 25, 30]
}

clf_adam_a2 = MLPClassifier()
clf_grid_adam_a2 = GridSearchCV(clf_adam_a2, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_adam_a2.fit(X_train, y_train)
preds_adam_a2 = clf_grid_adam_a2.predict(X_test)

clf_grid_adam_a2.best_params_


# In[86]:


print(classification_report(y_test, preds_adam_a2))


# In[87]:


clf_grid_adam_a2.cv_results_


# ### Activation (lbfs)

# In[88]:


parameters = {
    'solver': ['lbfgs'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [400, 500, 600], 
    'alpha': 10.0 ** -np.arange(7, 10), 
    'hidden_layer_sizes':[15, 20, 25]
}

clf_lbfs_a = MLPClassifier()
clf_grid_lbfs_a = GridSearchCV(clf_lbfs_a, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_lbfs_a.fit(X_train, y_train)
preds_lbfs_a = clf_grid_lbfs_a.predict(X_test)

clf_grid_lbfs_a.best_params_


# In[102]:


print(classification_report(y_test, preds_lbfs_a))


# In[103]:


clf_grid_lbfs_a.cv_results_


# ### Activation (lbfs) iter

# In[92]:


parameters = {
    'solver': ['lbfgs'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [500, 600, 700], 
    'alpha': 10.0 ** -np.arange(7, 10), 
    'hidden_layer_sizes':[15, 20, 25]
}

clf_lbfs_a_i = MLPClassifier()
clf_grid_lbfs_a_i = GridSearchCV(clf_lbfs_a_i, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_lbfs_a_i.fit(X_train, y_train)
preds_lbfs_a_i = clf_grid_lbfs_a_i.predict(X_test)

clf_grid_lbfs_a_i.best_params_


# In[93]:


print(classification_report(y_test, preds_lbfs_a_i))


# In[94]:


clf_grid_lbfs_a_i.cv_results_


# ### Activation (lbfs) iter2

# In[95]:


parameters = {
    'solver': ['lbfgs'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [700, 800, 900], 
    'alpha': 10.0 ** -np.arange(7, 10), 
    'hidden_layer_sizes':[15, 20, 25]
}

clf_lbfs_a_i2 = MLPClassifier()
clf_grid_lbfs_a_i2 = GridSearchCV(clf_lbfs_a_i2, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_lbfs_a_i2.fit(X_train, y_train)
preds_lbfs_a_i2 = clf_grid_lbfs_a_i2.predict(X_test)

clf_grid_lbfs_a_i2.best_params_


# In[97]:


print(classification_report(y_test, preds_lbfs_a_i2))


# In[98]:


clf_grid_lbfs_a_i2.cv_results_


# In[100]:


parameters = {
    'solver': ['lbfgs', 'sgd', 'adam'], 
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'max_iter': [700, 800, 900], 
    'alpha': 10.0 ** -np.arange(7, 10), 
    'hidden_layer_sizes':[15, 20, 25]
}

clf_h1 = MLPClassifier()
clf_grid_h1 = GridSearchCV(clf_h1, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_h1.fit(X_train, y_train)
preds_h1 = clf_grid_h1.predict(X_test)

clf_grid_h1.best_params_


# In[101]:


print(classification_report(y_test, preds_h1))


# ### Almost final (adam)

# In[104]:


parameters = {
    'solver': ['adam'], 
    'activation': ['logistic', 'tanh'],
    'max_iter': [700, 800, 900], 
    'alpha': 10.0 ** -np.arange(7, 10), 
    'hidden_layer_sizes':[20, 25, 30],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

clf_f1 = MLPClassifier()
clf_grid_f1 = GridSearchCV(clf_f1, parameters, cv=5, scoring="f1_micro", n_jobs=3)
clf_grid_f1.fit(X_train, y_train)
preds_f1 = clf_grid_f1.predict(X_test)

clf_grid_f1.best_params_


# In[ ]:





# In[105]:


print(classification_report(y_test, preds_f1))


# ### Almost final (lbfgs)

# In[24]:


parameters = {
    'solver': ['lbfgs'], 
    'activation': ['logistic'],
    'max_iter': [1500], 
    'alpha': 10.0 ** -np.arange(8, 9), 
    'hidden_layer_sizes':[20],
    'learning_rate': ['invscaling']
}

clf_f2 = MLPClassifier()
clf_grid_f2 = GridSearchCV(clf_f2, parameters, cv=5, scoring="f1_micro", n_jobs=3)
clf_grid_f2.fit(X_train, y_train)
preds_f2 = clf_grid_f2.predict(X_test)

clf_grid_f2.best_params_


# In[25]:


print(classification_report(y_test, preds_f2))


# ## sgd.

# In[21]:


parameters = {
    'solver': ['sgd'], 
    'max_iter': [400, 450, 500], 
    'alpha': 10.0 ** -np.arange(10, 11), 
    'hidden_layer_sizes':[70, 80, 90],
    'activation': ['identity', 'tanh', 'relu'],
    'learning_rate': ['adaptive']
}

clf_sgd_f_5 = MLPClassifier()
clf_grid_sgd_f_5 = GridSearchCV(clf_sgd_f_5, parameters, cv=5, scoring="f1_micro", n_jobs=-1)
clf_grid_sgd_f_5.fit(X_train, y_train)
preds_sgd_f_5 = clf_grid_sgd_f_5.predict(X_test)

clf_grid_sgd_f_5.best_params_


# In[22]:


print(classification_report(y_test, preds_sgd_f_5))


# Ο sgd δεν ξεπερνάει score μεγαλύτερο του 0.88, τον αφήνουμε.

# Παρόλο που παρακάμψαμε το βήμα του pipeline, βλέπουμε πως έχουμε ήδη αρκετά υψηλά scores (~0.94) στους adams και lbfgs με τις υπερπαραμέτρους στις οποίες καταλήξαμε.

# In[ ]:




