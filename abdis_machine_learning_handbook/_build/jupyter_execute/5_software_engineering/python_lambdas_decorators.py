#!/usr/bin/env python
# coding: utf-8

# # Lambdas & Decorators
# 
# ## Lets begin with Lambdas
# 
# No, not lambs. But, they're both pretty adorable
# 
# ![lamb kicking its feet](https://media0.giphy.com/media/9uIwrW53rQwNmHuGGq/giphy.gif)
# 
# In this notebook, we will aim to introduce the topic of lambda functions, and how they can be used.
# 
# -----

# ## What is a lambda function?
# A lambda function (aka anonymous functions, function literal, lambda abstractions) are function definitions that are not bound to an identifier. You can read more about what an anonymous function is [here](https://en.wikipedia.org/wiki/Anonymous_function).
# 
# ## why use Lambda functions?
# - when you want to have a very quick function that takes in abstract parameters
# - to create non-reuseble functions to stop there being tons of one-time use named functions 
# - for short-term use and doesn't need to be used
# 
# 
# ## why not use lambda functions?
# - they're not the only way to solve a problem e.g. you can create an actual function - named function
# - can make debugging hard, as python will only return that a lambda function has gone wrong....but which one, you'll never know!
# 

# ## Lambdas vs Functions
# 
# ![when you find lambdas, you never go back to functions...](https://pics.me.me/me-anormal-function-lambda-after-learning-lambdas-46113619.png)
# 
# Okay, so lambdas can be great. But what makes them different to normal functions?
# 
# - Lambdas only contain expressions and can't include statements in its body (e.g. set x to 5)
# - you can only write lambdas as a single line vs multiline functions
# - no statements like return, pass etc
# - 

# In[ ]:





# ## Examples
# 
# We will now write out examples in python. Why not copy these out and run them yourself?

# ### Examples of using Lambdas with Sorting

# In[1]:


unsort_count_of_a = ['aaaaaaaa', 'aa', 'aaaa', 'a']
unsort_count_of_a.sort(key=lambda item: len(item))
unsort_count_of_a


# ### Examples of Closures
# 
# Closueres are functions evaluated in environments that have bound variables. Okay, but what does that mean? Simply put, 

# In[ ]:





# Now, lets look at an example using `map()`.
# 
# Map takes two arguments: `function` to run and `iterables` to pass to the function.

# In[2]:


some_list = [1,3,5,7,8]
t = map(lambda i : i+1, some_list)
print(t)


# ## Okay, back to Decorators
# 
# What is a decorator?
# 
# A decorator simply wraps a function, and in turn changes its behaviour. Therefore, you can add additional behaviour to functions/classes. You will also see that you can use @ to call a decorator, instead of writing tons of code.
# 
# Here is an example

# In[3]:


def my_decorator(function):
    def wraps():
        print('wrapping the function...')
        function()
        print('wrapping complete.')
    return wraps

def print_hi():
    print('Hi abdi')

decorated = my_decorator(print_hi)
decorated()

