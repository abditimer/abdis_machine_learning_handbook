#!/usr/bin/env python
# coding: utf-8

# # Intro to Linear Algebra
# 
# 

# In[1]:


import numpy as np


# ## Matrix multiplication with NumPy
# 
# Element-wise multiplication

# In[2]:


# Element-wise multiplication
matrix_1 = np.array([
    [2,4,6],
    [1,2,3]
])
matrix_2 = np.array([
    [2,2,2],
    [3,3,3]
])
# is multiplying them with a * operator the same as np.multiply?
matrix_1 * matrix_2 == np.multiply(matrix_1, matrix_2)


# Matrix multiplication
# 
# Output matrix has:
# - same no of rows as the first matrix
# - same no of cols as the second matrix

# In[3]:


# (2,2) matrix
matrix_a = np.array([
    [1,1,1],
    [1,1,1]
])

# (2, 4)
matrix_b = np.array([
    [2,2],
    [2,2],
    [2,2]
])

np.matmul(matrix_a, matrix_b)


# Doing `Matmul` is essentially the same as doing the Dot product, if the matrices are 2D. That is why you will see it often.

# In[4]:


np.dot(matrix_a, matrix_b)

