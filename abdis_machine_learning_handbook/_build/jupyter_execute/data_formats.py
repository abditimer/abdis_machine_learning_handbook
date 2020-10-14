#!/usr/bin/env python
# coding: utf-8

# # Data - the different types
# 
# Data comes in different types, depending on your use case. For example, you may be analysing financial data relating to stocks, or looking at pictures your users have shared online on your app. No matter the type of data, it is your job, as a Machine Learning Engineer, to support the Data Engineers and Database Admins in your oganisation, to extract as much value out of this data as possible. This may be done by developing products, or supporting data scientists, analysts or other colleagues in the business.
# 
# ## Types of data
# 
# ### Structured Data
# Structured data is data that has a relationship, hence is *relational*. This means it has a strict schema which defines the structure (properties). Below, you can see an example of a table:
# 
# ![structured table image](https://lh3.googleusercontent.com/proxy/rQvqYgcqXK0XwSOTJqvniRagp-NHZY1REdVeMREcOANpgLie3jxIlst49DYBDGFmCzOVn_UfmWNO7evAii1TxR4Krq9PtcT7PSIfGrs)
# 
# Advantages of this is that it enables the data to have a structure that can be searched. However, this means that it may be difficult to change the data. For example, what if you have clients that share data with you in a variety of different ways? Having a rigid structure may mean you are unflexible to a change in demand.
# 
# 
# 
# 

# ### Semi-structured Data
# This sits between structured and unstructured data. This data uses tags to enable grouping/organisation of heirarchical data. It is where some structure exists.
# 
# This also is refered to as non-relational/NoSQL data. The structure will be described by a serialisation language - this can be used to write data, that is in memory, to something (an action e.g. read). Serialisation language examples include XML or JSON.
# 
# example of JSON:
# ```JSON
# {
#     "name": Abdi,
#     "how_awesome": 1000,
#     "age": Null
# }
# ```

# ### Unstructured data
# There is no structure - no seriously, there is some, but it can be varied. A good example here is an image, word document or even a song/voicenote. 

# ## Transactions
# 
# Wait, but what about if we are dealing with data that may be connected? E.g. if our picture is of a loan application that tells us about our customer?
# 
# In this case, we are able to group many smaller tasks/operations in a transaction. The benefit a transaction provides, is that if one trigger/event changes one piece of data, then the rest of the data is also changed. Also, this means if it fails, then it is able to roll back the entire series of operations.
# 
# E.g. if you look after a platform that looks after car insurance quotes, you can make sure with a transaction that an application is not completed until after the customer is underwritten.
# 
# But Abdi, what exactly is a transaction? It basically is a logical grouping of operations on a database that execute together.
# 
# If a change to one piece of data impacts another - you should be using transactions. They follow four requirements that are known as **ACID** guarantees. We have a friend that will step in here, feel free to ask it more detailed question if you need more help (visit him...[here](https://en.wikipedia.org/wiki/ACID) ;) :-
# * **A**tomicity: guarantees that each transaction is treated as a single unit, wich can either succeed completely or fail.
# * **C**onsistency: ensures a transaction can only bring the DB from one valid state to another.
# * **I**solation: ensures that concurrent execution of transactions leaves the DB in the same state that would have been obtained if the transaction were executed sequentially.
# * **D**urability: guarantees that once a transaction has been committed, it will remain committed even in the case of a system failure.
# 
# Thanks wikipedia for the help!
# 

# A lot of the time, data engineering does feel like: 
# ![funny data engineering image](https://miro.medium.com/max/1000/0*kAOakOPohpvdqM4M)
# 
# Transactional DBs, as described above, are referred to as OLTP (online transaction processing) systems. OLTP systems support:
# - many users
# - quick reads
# - handle large volumes of data
# - little downtime
# - handle simple and small transactions
# 
# OLAP (Online Analytical Processing) systems are the opposite. They support:
# - few users
# - long response times
# - more downtime
# - can handle large and complex transactions

# As a Machine Learning Engineer, it is important to understand when each of the varies different data formats may be appropriate. Hence, it is good to ask questions, such as the following, in order to get a better understanding of your data:
# * how often will the data be accessed? Is it accessed by customers who want instant information, or will business users use it, who do not mind waiting a little?
# * Is the data read-only? Should we only allow certain users to be able to write to the DB
# * etc...
