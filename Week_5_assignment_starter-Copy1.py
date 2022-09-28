#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment

# Using our prepared churn data from week 2:
# - use TPOT to find an ML algorithm that performs best on the data
#     - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
#     - REMEMBER: TPOT only finds the optimized processing pipeline and model. It doesn't create the model. 
#         - You can use `tpot.export('my_model_name.py')` (assuming you called your TPOT object tpot) and it will save a Python template with an example of the optimized pipeline. 
#         - Use the template code saved from the `export()` function in your program.
# - create a Python script/file/module using code from the exported template above that
#     - create a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
#     - your Python file/function should print out the predictions for new data (new_churn_data.csv)
#     - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# - test your Python module and function with the new data, new_churn_data.csv
# - write a short summary of the process and results at the end of this notebook
# - upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# 
# *Optional* challenges:
# - return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# - use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# - create a class in your Python module to hold the functions that you created
# - accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
# - Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# # Use TPOTT
# 	▪	Choose a metric you think is best to use for finding the best model

# In these next steps we will install TPOT and load the data.

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[6]:


pip install TPOT


# In[8]:


pip install XGBoost


# In[9]:


from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# In[10]:


import timeit 


# Now I'm going to load the data set

# In[12]:


df = pd.read_csv('prepped_telecom_data.csv', index_col='customerID')
df


# Next I will split it into features and labels

# In[16]:


features = df.drop('Churn', axis=1)
targets = df['Churn']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, test_size=0.2, random_state=42)


# Now we are using TPOTClassifer to find the accuracy score fo the pipeline

# In[24]:


get_ipython().run_cell_magic('time', '', 'tpot = TPOTClassifier(generations=5, population_size=100, verbosity=2, n_jobs=-1, random_state=42)\ntpot.fit(X_train, y_train)\nprint(tpot.score(X_test, y_test))\n\n')


# In[19]:


get_ipython().run_cell_magic('time', '', 'tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1, random_state=42)\ntpot.fit(X_train, y_train)\nprint(tpot.score(X_test, y_test))\n')


# Next, we are using TPOT Regressor

# In[21]:


from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# In[22]:


import timeit 


# In[23]:


get_ipython().run_cell_magic('time', '', "tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, n_jobs=-1, scoring='r2', random_state=42)\ntpot.fit(X_train, y_train)\nprint(tpot.score(X_test, y_test))")


# Based on the scores, it looks like TPOT classifier came up with a better model.

# Next I'm exporting the model.

# In[34]:


tpot.export('TPOTClassifierModel.py')


# # Create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
# 	▪	create a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
# 	▪	your Python file/function should print out the predictions for new data (new_churn_data.csv)
# 	▪	the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# 

# In[49]:


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('prepped_telecom_data.csv', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8037333333333333
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=MLPClassifier(alpha=0.001, learning_rate_init=0.1)),
    OneHotEncoder(minimum_fraction=0.25, sparse=False, threshold=10),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.05, min_samples_leaf=16, min_samples_split=19, n_estimators=100)
)

exported_pipeline.fit(X_train, y_train)

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)


# # Test your Python module and function with the new data, new_churn_data.csv

# In this step I'm setting the data and testing the module to see what it predicts.

# In[50]:


df = pd.read_csv('new_churn_data.csv', index_col='customerID')
df


# In[51]:


r = exported_pipeline.predict(df)
print (r)


# The model successfully returned the churn prediction for each row.

# # Summary

# In this week’s assignment, I first setup the TPOT model then based on the testing, I found out that TPOT classifier is the best model based on the accuracy score.
# 
# Then I created a Python module that returns the probability of churn for each row in the data frame. Next, I tested whether the model will work and I got some probabilities for each row of the new data.

# # Upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox

# ???
