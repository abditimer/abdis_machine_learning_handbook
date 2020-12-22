#!/usr/bin/env python
# coding: utf-8

# # Building a terrible diabetes predictor from scratch
# 

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


df = pd.read_csv('sentiment_data/diabetes.csv')
df.head(3)


# Lets build a function that removes the predicted class from our dataset & splits our data into the required format

# In[3]:


def clean_diabetes_data(data):
    data = data.drop('diab_pred', axis=1)
    target = data['diabetes']
    features = data.drop('diabetes', axis=1)
    return features, target


# In[4]:


features, target = clean_diabetes_data(df)


# In[5]:


X.head(3)


# In[32]:


len(X.to_numpy())


# In[13]:


y[:3]


# we will now build a class that will clean and train on this data

# In[92]:


import time
import sys 
from tqdm import trange 

class DiabetesPredictor(object):
    '''Build a network that predicts if you have diabetes

    Args:
        hidden_nodes (int) : no of hidden nodes we want
        learning_rate (float) : our learning rate

    '''
    def __init__(self, hidden_nodes=5, learning_rate=0.1):
        self.init_network(hidden_nodes, learning_rate)

    def clean_data(self, X):
        '''Returns normalised matrix
        '''
        # clean our X
        normalised_X = (X - X.mean()) / X.std()
        return normalised_X.to_numpy()

    def init_network(self, hidden_nodes, learning_rate):
        self.no_input = 7
        self.hidden_nodes = hidden_nodes
        self.no_output = 1
        self.learning_rate = learning_rate
        # init the input layer
        self.input_layer = np.zeros(shape=(1, 7))
        # init the weights
        self.w_0_1 = np.zeros(shape=(self.no_input, self.hidden_nodes))
        self.w_1_2 = np.random.normal(0.0, 1, size=(self.hidden_nodes, self.no_output))

        output_str = (
            '\n\nCreated a Neural Network with:\n'
            f'- {self.no_input} input nodes\n'
            f'- {self.hidden_nodes} hidden nodes\n'
            f'- {self.no_output} output nodes\n'
            '\n'
            'Our weights have the following shapes:\n'
            f'input to hidden: {self.w_0_1.shape}\n'
            f'hidden to output: {self.w_1_2.shape}\n'
        )
        print(output_str)

    def update_input_layer(self, row):
        self.input_layer *= 0
        for i in range(len(row)):
            self.input_layer[0][i] += row[i]

    def train(self, training_features, training_labels):
        assert(len(training_features) == len(training_labels))
        no_correct = 0
        start_time = time.time()

        self.cleaned_features = self.clean_data(training_features)

        for i in range(len(self.cleaned_features)):
            feature = self.cleaned_features[i]
            label = training_labels[i]

            self.update_input_layer(feature)

            ## Feedforward step
            layer_1_input = np.dot(self.input_layer, self.w_0_1)
            layer_1_output = layer_1_input # no activation function
            output_layer_input = np.dot(layer_1_output, self.w_1_2)
            output = self.sigmoid(output_layer_input)

            ## Backpropagate the error
            error = output - label
            layer_2_delta = error * self.sigmoid_prime(output)
            self.w_1_2 -= np.dot(layer_1_output.T, layer_2_delta)

            layer_1_error = np.dot(layer_2_delta, self.w_1_2.T) # propagate error even back
            layer_1_delta = layer_1_error
            self.w_0_1 -= np.dot(self.input_layer.T, layer_1_delta)

            if output >= 0.5 and label == 1:
                no_correct +=1
            elif output < 0.5 and label ==0:
                no_correct +=1

            elapsed_time = float(time.time() - start_time)
            reviews_per_s = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_features)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_s)[0:5]                              + " #Correct:" + str(no_correct) + " #Trained:" + str(i+1)                              + " Training Accuracy:" + str(no_correct * 100 / float(i+1))[:4] + "%")
            if(i % 1000 == 0):
                print("")

    def run(self, input):
        self.update_input_layer(input)
        pass

    def sigmoid(self, x):
        return 1 / (1+np.exp(x))
    
    def sigmoid_prime(self, x):
        return x * (1 - x)


# In[93]:


network = DiabetesPredictor()
network.train(features, target)

