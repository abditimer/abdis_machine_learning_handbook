#!/usr/bin/env python
# coding: utf-8

# # SKLearn API
# 

# 
# In this notebook, we will aim to go over the sklearn API, and how to use it to build your own ML models.
# 
# ## Why should I use SKLearn?
# Sklearn is one of pythons best implementation of machine learning algorithms and associated tools. The high level of the API is that it:
# - Consistency: All objects share a common interface drawn from a limited set of methods, with consistent documentation
# - Inspection: All specified parameter values are exposed as public attributes
# - Limited object hierarchy: Algos are represented as python classes; datasets are represented in NumPy/Pandas or SciPy; param names are string
# - Composition: allows you to express ML algorithms in a sequence of fundemental Algos
# - Sensible defaults: library defines appropriate default values
# 
# ## Sklearn Algos building blocks
# 
# They are made up of:
# - Estimators: The base object, implements a `fit` method to learn from data 
# - Predictor: uses `predict` or `predict_proba`
# - Transformer: for filtering or modifying data
# - Model: A model that can give a goodness of fit measure or a likelihood of unseen data 

# In[1]:


5+5

