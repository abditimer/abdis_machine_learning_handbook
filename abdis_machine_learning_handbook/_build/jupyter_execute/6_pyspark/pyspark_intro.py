#!/usr/bin/env python
# coding: utf-8

# # PySpark refresher
# 
# ![spark image](https://databricks.com/wp-content/uploads/2019/02/largest-open-source-apache-spark.png)
# 
# This notebook will introduce you to the topic of pyspark.
# 
# 

# In[1]:


5+5


# ## Background
# 
# ### What is spark and why should I care?
# 
# At a high-level, spark is:
# - open-source distributed querting and processing  engine. It is written in Scala and Java, running in JVM
# - it is very fast (compared to hadoop)
# - API access via Scala, python and more
# 
# Spark, at a high level, can be viewed as:
# 
# ![spark ecosystem overview](https://docs.snowflake.com/en/_images/spark-snowflake-data-source.png)
# 
# From the image above, you can see that spark is made up of a core API that abstracts the scala and java code. It is made up of:
# - spark SQL: more efficient way of interacting with data 
# - spark streaming, mllib, graphx: streaming, graph and ML frameworks
# 
# ### But how does spark work?
# 
# You can think of spark of being made up of `Jobs`. These jobs must have atleast one `Driver` and one `Worker`: A Driver plans all the execution of the jobs and commands the workers to carry out a specific `task` from the job. A worker does not decide what task they want to run, instead, this is commanded by the driver. Hence, we can summarise a spark job as a optimised set of tasks that are processed in order. Spark is able to optimise this by identifying which tasks can be run in parallel. This can also be viewed by the generated Directed Acyclic Graph (DAG) that is created by spark.
