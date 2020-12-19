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