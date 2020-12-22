#!/usr/bin/env python
# coding: utf-8

# # Basics of Deep Learning
# In this notebook, we will cover the basics behind Deep Learning. I'm talking about building a brain....
# 
# ![gif of some colours](https://www.fleetscience.org/sites/default/files/images/neural-mlblog.gif)
# 
# Only kidding. Deep learning is a fascinating new field that has exploded over the last few years. From being used as facial recognition in apps such as Snapchat or challenger banks, to more advanced use cases such as being used in [protein-folding](https://www.independent.co.uk/life-style/gadgets-and-tech/protein-folding-ai-deepmind-google-cancer-covid-b1764008.html).
# 
# In this notebook we will:
# - Explain the building blocks of neural networks
# - Go over some applications of Deep Learning

# ## Building blocks of Neural Networks
# 
# I have no doubt that you have heard/seen how similar neural networks are to....our brains. 
# 
# 
# ### The Perceptron
# 
# The building block of neural networks. The perceptron has a rich history (covered in the background section of this book). The perceptron was created in 1958 by Frank Rosenblatt (I love that name) in Cornell, however, that story is for another day....or section in this book (backgrounds!),
# 
# The perceptron is an algorithm that can learn a binary classifier (e.g. is that a cat or dog?). This is known as a threshold function, which maps an input vector *x* to an output decision *$f(x)$ = output*. Here is the formal maths to better explain my verbal fluff:
# 
# $ f(x) = { 1 (if: w.x+b > 0), 0 (otherwise) $
# 

# ### The Artificial neural network
# 
# Lets take a look at the high level architecture first.
# 
# ![3blue1brown neural network gif](https://thumbs.gfycat.com/DeadlyDeafeningAtlanticblackgoby-max-1mb.gif) 
# 
# The gif above of a neural network classifying images is one of the best visual ways of understanding how neural networks, work. The neural network is made up of a few key concepts:
# - An input: this is the data you pass into the network. For example, data relating to a customer (e.g. height, weight etc) or the pixels of an image
# - An output: this is the prediction of the neural network
# - A hidden layer: more on this later
# - Neuron: the network is made up of neurons, that take an input, and give an output
# 
# Now, we have a slightly better understanding of what a neuron is. Lets look at a very simple neuron:
# 
# ![simple neural network](https://databricks.com/wp-content/uploads/2019/02/neural1.jpg) 
# 
# From the above image, you can clearly see the three components listed above together. 
# 
# ### But Abdi, what is the goal of a neural network?
# 
# Isn't it obvious? To me, it definitely was not when I first started to learn about neural networks. Neural networks are beautifully complex to understand, but with enough time and lots of youtube videos, you'll be able to master this topic.
# 
# The goal of a neural network is to make a pretty good guess of something. For example, a phone may have a face unlock feature. The phone probably got you to take a short video/images of yourself in order to set up this security feature, and when it **learned** your face, you were then able to use it to unlock your phone. This is pretty much what we do with neural networks. We teach it by giving it data, and making sure it gets better at making predictions by adjusting the weights between neurons. More on this soon.
# 

# ## Gradient Descent Algo
# 
# One of the best videos on neural networks, by 3Blue1Brown:
# 
# <figure class="video_container">
#   <iframe src="https://www.youtube.com/watch?v=aircAruvnKk" frameborder="0" allowfullscreen="true"> </iframe>
# </figure>
# 
# His series on Neural networks and Linear algebra are golden sources for learning Deep Learning.

# ### Simple Gradient Descent Implementation
# with the help from our friends over at Udacity, please view below an implementation of the Gradient Descent Algo. This is a very basic neural network that only has its inputs linked directly to the outputs.
# 
# We begin by defining some functions.

# In[1]:


import numpy as np 

# We will be using a sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivation of sigmoid(x) - will be used for backpropagating errors through the network
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


# We begin by defining a simple neural network:
# - two input neurons: x1 and x2
# - one output neuron: y1

# In[2]:


x = np.array([1,5])
y = 0.4


# We now define the weights, w1 and w2 for the two input neurons; x1 and x2. Also, we define a learning rate that will help us control our gradient descent step

# In[3]:


weights = np.array([-0.2,0.4])
learnrate = 0.5


# we now start moving forwards through the network, known as feed forward. We can combine the input vector with the weight vector using numpy's dot product

# In[4]:


# linear combination
# h = x[0]*weights[0] + x[1]*weights[1]
h = np.dot(x, weights)


# We now apply our non-linearity, this will provide us with our output.

# In[5]:


# apply non-linearity
output = sigmoid(h)


# Now that we have our prediction, we are able to determine the error of our neural network. Here, we will use the difference between our actual and predicted.

# In[6]:


error = y - output


# The goal now is to determine how to change our weights in order to reduce the error above. This is where our good friend gradient descent and the chain rule come into play:
# - we determine the derivative of our error with respect to our input weights. Hence:
# - change in weights = $ \frac{d}{dw_{i}} \frac{1}{2}{(y - \hat{y})^2}$
# - simplifies to = learning rate * error term * $ x_{i}$
# - where:
#     - learning rate = $ n $
#     - error term = $ (y - \hat{y}) * f'(h) $
#     - h =  $ \sum_{i} W_{i} x_{i} $ 
# 
# 
# We begin by calculating our f'(h)

# In[7]:


# output gradient - derivative of activation function
output_gradient = sigmoid_prime(h)


# Now, we can calcualte our error term

# In[8]:


error_trm = error * output_gradient


# With that, we can update our weights by combining the error term, learning rate and our x

# In[9]:


#gradient desc step - updating the weights
dsc_step = [
    learnrate * error_trm * x[0],
    learnrate * error_trm * x[1]
]


# Which leaves...

# In[10]:


print(f'Actual: {y}')
print(f'NN output: {output}')
print(f'Error: {error}')
print(f'Weight change: {dsc_step}')


# ### More in depth...
# 
# Lets now build our own end to end example. we will begin by creating some fake data, followed by implementing our neural network.

# In[11]:


x = np.random.rand(200,2)
y = np.random.randint(low=0, high=2, size=(200,1))


# In[12]:


no_data_points, no_features = x.shape


# In[13]:


def sig(x):
    '''Calc for sigmoid'''
    return 1 / (1+np.exp(-x))


# In[14]:


weights = np.random.normal(scale=1/no_features**.5, size=no_features)


# In[15]:


epochs = 1000
learning_rate = 0.5


# In[16]:


last_loss = None

for single_data_pass in range(epochs):
    # Creating a weight change tracker
    change_in_weights = np.zeros(weights.shape)
    for x_i, y_i in zip(x, y):
        h = np.dot(x_i, weights)
        y_hat = sigmoid(h)
        error = y_i - y_hat
        # error term = error * f'(h)
        error_term = error * (y_hat * (1-y_hat))
        # now multiply this by the current x & add to our weight update
        change_in_weights += (error_term * x_i)
    # now update the actual weights
    weights += (learning_rate * change_in_weights / no_data_points)

    # print the loss every 100th pass
    if single_data_pass % (epochs/10) == 0:
        # use current weights in NN to determine outputs
        output = sigmoid(np.dot(x_i,weights))
        # find the loss
        loss = np.mean((output-y_i)**2)
        # 
        if last_loss and last_loss < loss:
            print(f'Train loss: {loss}, WARNING - Loss is inscreasing')
        else:
            print(f'Training loss: {loss}')
        last_loss = loss 


# ## Multilayer NN
# 
# Now, lets build upon our neural network, but this time, we have a hidden layer.
# 
# Lets first see how to build the network to make predictions.

# In[17]:


X = np.random.randn(4)
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(4, 3))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(3, 2))


# In[18]:


sum_input = np.dot(X, weights_input_to_hidden)
h = sigmoid(sum_input)

sum_h = np.dot(h, weights_hidden_to_output)
y_pred = sigmoid(sum_h)


# ## Backpropa what?
# 
# Ok, so now, how do we refine our weights? Well, this is where **backpropagation** comes in. After feeding our data forwards through the network, using feed forward, we propagate our errors backwards, making use of things such as the chain rule.
# 
# Lets do an implementation.

# In[19]:


# we have three input nodes
x = np.array([0.5, 0.2, -0.3])
# one output node
y = 0.7

learnrate = 0.5
# 2 nodes in hidden layer
weights_input_hidden = np.array(
    [
        [0.5, -0.6], [0.1, -0.2], [0.1, 0.7]
    ]
)
weights_hidden_output = np.array([
    0.1,-0.3
])


# In[20]:


# feeding data forwards through the network
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
#---
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
y_hat = sigmoid(output_layer_input)


# In[21]:


# backward propagate the errors to tune the weights

# 1. calculate errors
error = y - y_hat
output_node_error_term = error * (y_hat * (1-y_hat))
#----
hidden_node_error_term = weights_hidden_output * output_node_error_term *(hidden_layer_output * (1-hidden_layer_output))

# 2. calculate weight changes
delta_w_output_node = learnrate * output_node_error_term * hidden_layer_output
#-----
delta_w_hidden_node = learnrate * hidden_node_error_term * x[:,None]


# In[22]:


print(f'Original weights:\n{weights_input_hidden}\n{weights_hidden_output}')
print()
print('Change in weights for hidden layer to output layer:')
print(delta_w_output_node)
print('Change in weights for input layer to hidden layer:')
print(delta_w_hidden_node)


# ## Putting it all together

# In[23]:


features = np.random.rand(200,2)
target = np.random.randint(low=0, high=2, size=(200,1))

def complete_backprop(x,y):
    '''Complete implementation of backpropagation'''
    n_hidden_units = 2
    epochs = 900
    learnrate = 0.005

    n_records, n_features = features.shape
    last_loss = None

    w_input_to_hidden = np.random.normal(scale=1/n_features**.5,size=(n_features, n_hidden_units))
    w_hidden_to_output = np.random.normal(scale=1/n_features**.5, size=n_hidden_units)

    for single_epoch in range(epochs):
        delw_input_to_hidden = np.zeros(w_input_to_hidden.shape)
        delw_hidden_to_output = np.zeros(w_hidden_to_output.shape)

        for x,y in zip(features, target):
            # ----------------------
            # 1. Feed data forwards
            # ----------------------
            
            hidden_layer_input = np.dot(x,w_input_to_hidden)
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, w_hidden_to_output)
            output_layer_output = sigmoid(output_layer_input)

            # ----------------------
            # 2. Backpropagate the errors
            # ----------------------

            # error at output layer
            prediction_error = y - output_layer_output
            output_error_term = prediction_error * (output_layer_output * (1-output_layer_output))

            # error at hidden layer (propagated from output layer)
            # scale error from output layer by weights
            hidden_layer_error = np.multiply(output_error_term, w_hidden_to_output)
            hidden_error_term = hidden_layer_error * (hidden_layer_output * (1-hidden_layer_output))

            # ----------------------
            # 3. Find change of weights for each data point
            # ----------------------

            delw_hidden_to_output += output_error_term * hidden_layer_output
            delw_input_to_hidden += hidden_error_term * x[:,None]
        
        
        # Now update the actual weights
        w_hidden_to_output += learnrate * delw_hidden_to_output / n_records
        w_input_to_hidden += learnrate * delw_input_to_hidden / n_records

        # Printing out the mean square error on the training set
        if single_epoch % (epochs / 10) == 0:
            hidden_output = sigmoid(np.dot(x, w_input_to_hidden))
            out = sigmoid(np.dot(hidden_output,
                                w_hidden_to_output))
            loss = np.mean((out - target) ** 2)

            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss


complete_backprop(features,target)

