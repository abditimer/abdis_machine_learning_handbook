import time
import sys 
import numpy as np 
from tqdm import trange

class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        '''Neural Network for sentiment analysis
        
        Args:
            reviews (list) : reviews used for training
            labels (list) : labels for each review used for training
            hidden_nodes (int) : number of nodes in our hidden layer
            learning_rate (float) : learning rate used for backprop
        '''
        np.random.seed(1)

        # process the reviews & labels
        self.data_ingestion(reviews, labels)
        # build the network
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def data_ingestion(self, reviews, labels):
        '''Preprocess our data'''

        print('processing input...')

        review_vocab = set()
        label_vocab = set()
        for i in range(len(reviews)):
            for word in reviews[i].split(' '):
                review_vocab.add(word)
            label_vocab.add(labels[i])
        self.review_vocab = list(review_vocab)
        self.label_vocab = list(label_vocab)
        
        # remember the size
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # create a mapping for the words in our vocabs
        self.word2index = {}
        self.label2index = {}
        for i in trange(self.review_vocab_size):
            self.word2index[self.review_vocab[i]] = i
        for i in trange(self.label_vocab_size):
            self.label2index[self.label_vocab[i]] = i

    def init_network(self, no_input_nodes, no_hidden_nodes, no_output_nodes, learning_rate):
        self.no_input_nodes = no_input_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.no_output_nodes = no_output_nodes
        self.learning_rate = learning_rate

        # initialise input layer
        self.input_layer = np.zeros(shape=(1, self.no_input_nodes))

        # initialise the weights
        self.w_0_1 = np.zeros(shape=(self.no_input_nodes, self.no_hidden_nodes))
        self.w_1_2 = np.random.normal(0.0, 1, size=(self.no_hidden_nodes, self.no_output_nodes))

        output_str = (
            '\n\nCreated a Neural Network with:\n'
            f'- {self.no_input_nodes} input nodes\n'
            f'- {self.no_hidden_nodes} hidden nodes\n'
            f'- {self.no_output_nodes} output nodes\n'
            '\n'
            'Our weights have the following shapes:\n'
            f'input to hidden: {self.w_0_1.shape}\n'
            f'hidden to output: {self.w_1_2.shape}\n'
        )
        print(output_str)

    def update_input_layer(self, review):
        # clear previous input layer
        self.input_layer *= 0
        # add each word from the review to the input layer
        for word in review.split(' '):
            if word in self.word2index:
                # set the position of that word in our input layer to 1
                self.input_layer[0][self.word2index[word]] += 1

    def train(self, training_reviews, training_labels):
        assert(len(training_reviews) == len(training_labels))
        correct_so_far = 0
        start_time = time.time()

        for i in range(len(training_reviews)):
            review = training_reviews[i]
            label = training_labels[i]
            y_actual = self.translate_label(label)

            # Feed our input forwards
            self.update_input_layer(review) # set input layer
            layer_1_input = np.dot(self.input_layer, self.w_0_1) 
            layer_1_output = layer_1_input # layer 1 output (no non-linearity)
            layer_2_input = np.dot(layer_1_output, self.w_1_2) 
            layer_2_output = self.sigmoid(layer_2_input)

            # Back propagate the error
            layer_2_error = layer_2_output - y_actual
            layer_2_delta = layer_2_error * self.sigmoid_prime(layer_2_output)
            
            layer_1_error = np.dot(layer_2_delta, self.w_1_2.T) 
            layer_1_delta = layer_1_error # there is no non-linearity

            self.w_1_2 -= np.dot(layer_1_output.T, layer_2_delta) * self.learning_rate
            self.w_0_1 -= np.dot(self.input_layer.T, layer_1_delta) * self.learning_rate

            if (layer_2_output >= 0.5 and label =='POSITIVE'):
                correct_so_far +=1
            elif (layer_2_output < 0.5 and label =='NEGATIVE'):
                correct_so_far +=1

            elapsed_time = float(time.time() - start_time)
            reviews_per_s = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_s)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        correct_preds = 0
        start_time = time.time()
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct_preds += 1

            elapsed_time = float(time.time() - start_time)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_preds) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct_preds * 100 / float(i+1))[:4] + "%")

    def run(self, review):
        self.update_input_layer(review.lower())
        layer_1 = np.dot(self.input_layer, self.w_0_1)
        layer_2 = np.dot(layer_1, self.w_1_2)
        y_pred = self.sigmoid(layer_2)

        if y_pred[0] >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def sigmoid_prime(self, x):
        return x * (1-x)

    def translate_label(self, label):
        if label == 'POSITIVE':
            return 1
        else: 
            return 0

##--------------------------------------------
##############################################
##--------------------------------------------
from collections import Counter 
class NewNetwork:
    def __init__(self, file_folder, no_hidden=10, lrate=0.1):
        '''Improved network.
        Args:
            file_folder (str) : folder where the data is stored
            network_structure (list) : list of [input_nodes, hidden_nodes, output_nodes]
        '''
        # process data
        self.read_file(file_folder)
        self.process_data() 
        # define network
        self.init_network(no_hidden, lrate)
        self.print_pretty_network_structure()

    def init_network(self, no_hidden, lrate):
        '''Build our architecture for the network

        Args:
            no_hidden (int) : number of hidden nodes
            lrate (float) : number for our learning rate
        '''
        self.no_input_nodes = len(self.review_vocab)
        self.no_hidden_nodes = no_hidden
        self.no_output_nodes = 1

        self.learning_rate = lrate
        self.input_layer = np.zeros(shape=(1, self.no_input_nodes))

        self.w_input_hidden = np.zeros(shape=(self.no_input_nodes, self.no_hidden_nodes))
        self.w_hidden_ouput = np.random.normal(0.0, 1, size=(self.no_hidden_nodes, self.no_output_nodes))

    def train(self, training_reviews, training_labels, epochs=2):
        correct = 0
        starting_time = time.time()

        for i in range(len(training_reviews)):
            x = training_reviews[i]
            y = self.translate_label(training_labels[i])
            # update input layer
            self.update_input_layer(x)
            
            # define forward pass
            hidden_output = np.dot(self.input_layer, self.w_input_hidden) # no activation
            output = self.sigmoid(np.dot(hidden_output, self.w_hidden_ouput))
            
            # define backwards pass
            error = y - output 
            delta_l2 = error * self.sigmoid_prime(output)
            error_l1 = np.dot(delta_l2, self.w_hidden_ouput.T)
            delta_l1 = error_l1
            # update weights
            self.w_hidden_ouput -= self.learning_rate * np.dot(hidden_output.T, delta_l2)
            self.w_input_hidden -= self.learning_rate * np.dot(self.input_layer.T, delta_l1)
            
            # print continual loss
            if output >= 0.5 and y == 'POSITIVE':
                correct +=1
            elif output < 0.5 and y == 'NEGATIVE':
                correct +=1
            
            time_diff = float(time.time() - starting_time)
            reviews_per_second = i / time_diff if time_diff > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, training_reviews):
        pass

    def run(self, single_review):
        '''Run a review through our network to get a prediction.
        Args: single_review (str) : str sentene review
        '''
        # feed into input_layer
        self.update_input_layer(single_review)

        # feed forward pass
        x = np.dot(self.input_layer, self.w_input_hidden)
        x = np.dot(x, self.w_hidden_ouput)
        pred = self.sigmoid(x) 

        # return translated prediction
        self.translate_prediction(pred)

    def update_input_layer(self, single_review):
        # reset the current input layer
        self.input_layer *= 0
        
        for word in single_review.split(' '):
            if word in self.vocab_word_to_index['review']:
                index_of_word = self.vocab_word_to_index['review'][word]
                self.input_layer[0][index_of_word] = 1 

    def process_data(self):
        '''Process data by creating vocabulary and index
        '''
        self.review_vocab, self.label_vocab = self.create_vocab()
        self.vocab_word_to_index = self.create_indexer()
    
    def create_vocab(self):
        '''Create vocabulary from self.train and self.label
        Returns:
            review_vocab (list) : unique words from all our reviews
            label_vocab (list) : unique words from all our labels 
        '''
        review_vocab = set()
        for i in range(len(self.reviews)):
            for word in self.reviews[i].split(' '):
                review_vocab.add(word)
        review_vocab = list(review_vocab)
        
        label_vocab = set()
        for i in range(len(self.labels)):
            label_vocab.add(self.labels[i])
        label_vocab = list(label_vocab)

        return review_vocab, label_vocab

    def create_indexer(self):
        '''Create an index for each word.
        
        Returns:
            vocab_word_to_index (dict) : maps a word to an index position
        '''
        vocab_word_to_index = {}
        vocab_word_to_index.update({'review' : {}, 'label': {}})

        for i in range(len(self.review_vocab)):
            vocab_word_to_index['review'][self.review_vocab[i]] = i 
        
        for j in range(len(self.label_vocab)):
            vocab_word_to_index['label'][self.label_vocab[j]] = j

        return vocab_word_to_index

    def read_file(self, file_folder):
        '''Read file from folder location.
        
        Args:
            file_folder (str): folder location with review and label files    
        '''
        review_file_local = file_folder + '/reviews.txt'
        label_file_local = file_folder + '/labels.txt'
        try:
            print('Attempting to read files...')
            review_file = open(review_file_local, 'r')
            self.reviews = list(map(lambda x : x[:-1], review_file.readlines()))
            review_file.close()

            labels_file = open(label_file_local, 'r')
            self.labels  = list(map(lambda x : x[:-1].upper(), labels_file.readlines())) 
            labels_file.close()
        except:
            raise Exception('Please store review and label files in the folder.')
        else:
            print('Files read.')
            print('Example - first two reviews:')
            print(self.labels[0] + '\t:\t' + self.reviews[0][:80] + '...')
            print(self.labels[1] + '\t:\t' + self.reviews[1][:80] + '...')

    def print_pretty_network_structure(self):
        output_str = (
            '\n\nCreated a Neural Network with:\n'
            f'- {self.no_input_nodes} input nodes\n'
            f'- {self.no_hidden_nodes} hidden nodes\n'
            f'- {self.no_output_nodes} output nodes\n'
            '\n'
            'Our weights have the following shapes:\n'
            f'input to hidden: {self.w_input_hidden.shape}\n'
            f'hidden to output: {self.w_hidden_ouput.shape}\n'
            '\n'
            f'Lastly, with a Learning rate of {self.learning_rate}'
        )
        print(output_str) 

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    
    def sigmoid_prime(self, output):
        return output * ( 1 - output)

    def translate_prediction(self, x):
        print('POSITIVE' if x >= 1 else 'NEGATIVE')

    def translate_label(self, x):
        return 1 if x == 'POSITIVE' else 0