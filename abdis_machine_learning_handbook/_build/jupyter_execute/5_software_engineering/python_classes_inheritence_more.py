#!/usr/bin/env python
# coding: utf-8

# # Python Classes, Inheritence etc
# 
# This notebook will include some basics of software engineering.
# 
# We will try to cover both theoretical and practical concepts. 

# In[5]:


class test:
    def __init__(self):
        print('always runs...')
    def maybe_run(self):
        print('running now that you\'ve called me.')

test  = test()


# In[6]:


test.maybe_run()

