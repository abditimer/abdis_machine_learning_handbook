#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis end-to-end example
# 
# This example is brought to you by Udacity - consider doing the great Udacity Deep Learning course. Find out more [here](https://www.udacity.com/course/deep-learning-nanodegree--nd101). 
# 
# > These are my own personal notes
# 
# ----

# In this notebook, the aium is to build `TODO`
# 
# We begin by looking at the dataset we have:
# - reviews.txt: reviews of a movie
# - labels.txt: positive/negative label associated with the movie 
# 
# We will use the python `open()` function to open the file, with the parameter `'r'` to read the file. Using `readlines` will return a list made up of each line in the file, returned as a list item. Hence, each character will be an item in the list. 

# In[1]:


review_file = open('sentiment_data/reviews.txt', 'r')
reviews = list(map(lambda x : x[:-1], review_file.readlines()))
review_file.close()

label_file = open('sentiment_data/labels.txt', 'r')
labels = list(map(lambda x : x[:-1].upper(), label_file.readlines()))
label_file.close()


# Now lets find some information out about our data.

# In[2]:


print(f'Size of our data: {len(reviews)}')
print(f'No of labels: {len(labels)}')
print('\nNow, lets see one row of our data. First feature in our data:')
print(reviews[0])
print('\nPrediction:')
print(labels[0])


# ---

# ## Now, lets build up a hypothesis
# 
# We will begin by looking at our data, and trying to see what conclusions we can draw. This is ofter called the `exploratory` phase. We will begin by looking at some random predictions...

# In[3]:


def print_review_with_label(ith_row):
    print(labels[ith_row] + '\t:\t' + reviews[ith_row][:80] + '...')


# Using the function above, we can beautifully print our data; feature along with its prediction.

# In[4]:


print("labels.txt \t : \t reviews.txt\n")
print_review_with_label(2137)
print_review_with_label(12816)
print_review_with_label(6267)
print_review_with_label(21934)


# We will be using the `Counter` python class throughout this section, as it provides a nice way to count the occurances of words. 

# In[5]:


from collections import Counter
import numpy as np 


# In[6]:


positive_words_counter = Counter()
negative_words_counter = Counter()
total_words_counter = Counter()
example_counter_with_stuff = Counter([1,2,3,4,4,4])

def counter_pretty_print():
    print('positive counter: ', positive_words_counter)
    print('negative counter: ', negative_words_counter)
    print('total words counter: ', total_words_counter)

print('At this stage, our counters are empty...')
counter_pretty_print()
print('Here is a test counter: ', example_counter_with_stuff)


# Now, lets fill out our three counters.

# In[7]:


# for each row in our dataset
for sentence_no in range(len(reviews)):
   # for each word in our sentence
   for word in reviews[sentence_no].split(' '):
       # if it is positive - add a positive counter
       if labels[sentence_no] == 'POSITIVE':
           positive_words_counter[word] +=1
       # if it is negative - add to negative counter
       if labels[sentence_no] == 'NEGATIVE':
           negative_words_counter[word] +=1
       # regardless, add to total word counter
       total_words_counter[word] +=1


# In[8]:


# lets take a look at the most common words.
print('Most common positive words:\n')
positive_words_counter.most_common()


# In[9]:


# lets take a look at the most common words.
print('\nMost common negative words:\n')
negative_words_counter.most_common()


# Instead of looking at the counts of the words, lets now instead look at the ratios between words. Looking at how often words occur, either positive or negative, does not really give us what we are looking for. e.g. you can see there are a lot of common words between both the positive and negative counters. Instead, by looking at a raio, we will be looking at the words that are found in positive reviews over negative, and vice versa. 
# 
# This will basically tell us how many more times a word is seen in positive reviews than in the negatives. e.g. we can imagine that positive reviews use the word "love" more, hence the ratio should be larger. Hence:
# - Positive words will have a large ratio - bigger than 1
# - Negative words will have a smaller ratio - less than 1
# - words that are neither positive or negative, but neutral, will be centered around 0

# In[10]:


positive_to_negative_ratio = Counter()

for word, count, in list(total_words_counter.most_common()):
    if count > 100:
        positive_to_negative_ratio[word] = positive_words_counter[word] / (negative_words_counter[word] + 1) # +1 so we dont divide by 0


# Now, lets take a look at some words...

# In[11]:


print(f'positive:negative ratio for the word and: {round(positive_to_negative_ratio["and"],2)}')
print(f'positive:negative ratio for the word good: {round(positive_to_negative_ratio["best"],2)}')
print(f'positive:negative ratio for the word bad: {round(positive_to_negative_ratio["bad"],2)}')


# Okay, but is a score of 2 twice as good as other scores? With the ratios as they are now, it will be difficult to actually compare the scores. So instead, we will do what every computer scientists loves to do, which is to log the numbers.
# 
# To find out more about why computer scientists love log, feel free to watch the series by Killian Weiberger on Machine Learning [here](https://www.youtube.com/watch?v=MrLPzBxG95I&list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS)

# In[12]:


for word, count in positive_to_negative_ratio.most_common():
    positive_to_negative_ratio[word] = np.log(count)


# Now, lets take a look at the log(words)...

# In[13]:


print(f'positive:negative ratio for the word and: {round(positive_to_negative_ratio["and"],2)}')
print(f'positive:negative ratio for the word good: {round(positive_to_negative_ratio["best"],2)}')
print(f'positive:negative ratio for the word bad: {round(positive_to_negative_ratio["bad"],2)}')


# You can see now that:
# - positive words are close to +1
# - negative words are close to -1
# - neutral words are centered around 0
# 
# Now, to close our hypothesis section where we wanted to draw a hypothesis from the data, we will take a peek at our ratio data.

# In[14]:


positive_to_negative_ratio.most_common()[0:20]


# As we expected, some positive words like `flawless` and `perfection` have high scores....but also `lincoln`. Interesting.

# In[15]:


list(reversed(positive_to_negative_ratio.most_common()))[0:20]


# There are some funny negative words, including `lousy` and `unwatcheable`. But again, some interesting words like `prom`.

# ## Transforming words into numbers
# 
# we now need to prerpare our words so that we can feed them into our neural network. In order to do that, we want to transform them so we can do the maths of neural networks.
# 
# What we want to do for our network, is build a dictionary. With this dictionary, we will count each word in our input review, and feed that into the network.
# 
# As we have already built a Count object that has every word possible from our training data, we are able to now compare each single review from our dataset, and see how often each word occurs per review. This will allow us to feed our reviews into the network whilst maintaining consistency between inputs.
# 
# We will begin by building a `vocab`, a set that contains all the words.

# In[16]:


vocab = set(total_words_counter.keys())


# Vocab is s Set, similar to the mathematical set. This means that it only has each word appearing only once. 
# 
# Now, lets take a look at how our Neural network will look. 
# 
# ![image of our neural network](sentiment_network.png)
# 
# You can see that our NN will have:
# - one input layer: 
#     - This will be the Vocab
#     - we will represent this as a np array
# - one hidden layer
# - one output layer that has one output neuron

# In[17]:


layer_0 = np.zeros(shape=(1,len(vocab)))


# lets take a look at the first layer...

# In[18]:


layer_0.shape


# This first layer now has a neuron/input per word from our vocab. With the input being a count of how many times the word occurs in the review. However, to pass words from a review into this first layer, we need to be able to build a way that will allow us to feed a new review in with the words organised the same way as the first layer in our network.

# In[19]:


word_to_index_translator = {}
# lets map each word in our vocab to an index, and capture that as a dictionary
for index, word in enumerate(vocab):
    word_to_index_translator[word] = index

# lets temporarily use a Counter object to look at the first few rows in our dictionary
Counter(word_to_index_translator).most_common(5)


# now, lets build a function that can take a new review, and spit out a vector that matches the input layer.

# In[20]:


def input_for_input_layer(review):
    ''' New input layer, layer_0, for our network to train on.

    layer_0 represents how many times a word occurs in a review.

    Args:
        review (str) : a review for a movie
    Returns:
        None
    '''
    global layer_0
    # clear out previous layer 0
    layer_0 *=0
    for word in review.split(' '):
        # find index location of the word from our vocab
        index_of_word = word_to_index_translator[word]
        # add it to our layer 0
        layer_0[:, index_of_word] += 1


# Lets test this by feeding it a review.
# Before we test it, lets look at layer_0

# In[21]:


layer_0


# In[22]:


input_for_input_layer(reviews[200])
layer_0


# Great, it has updated layer_0.
# 
# Now, we will build a function that can take a label (e.g. POSITIVE or NEGATIVE), and return either 1 or 0. This is needed as our network needs to be built ontop of numbers, and not strings.

# In[23]:


def translate_label(label):
    '''Converts label to 0 or 1.

    Args:
        label (str) : POSITIVE or NEGATIVE label for a review
    RETURNS:
        0 : if negative
        1 : if positive
    '''
    if label == 'POSITIVE':
        return 1
    else:
        return 0


# again, lets test this by running a label into our function.

# In[24]:


print(f'testing +ve label: {labels[200]}')
print(f'This is the output from our function: {translate_label(labels[200])}')
print(f'\ntesting -ve label: {labels[1]}')
print(f'This is the output from our function: {translate_label(labels[1])}')


# Great, so it works.
# 
# Now it is finally time to build our Neural Network!
# 
# We will:
# - build a basic neural network that has an input layer, hidden layer and an output layer
# - we will not be adding non-linearity in our hidden layer
# - we will use the same functions we defined above, to build up our training data set
# - we will create a vocab from our training data
# - we will train over the entire corpus

# In[25]:


import sentiment_network
import importlib
importlib.reload(sentiment_network)


# In[113]:


mlp = sentiment_network.SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)


# In[114]:


mlp.test(reviews[-1000:],labels[-1000:])


# In[115]:


mlp.train(reviews[:-1000],labels[:-1000])


# In[130]:


mlp2 = sentiment_network.SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
mlp2.train(reviews[:-1000],labels[:-1000])

