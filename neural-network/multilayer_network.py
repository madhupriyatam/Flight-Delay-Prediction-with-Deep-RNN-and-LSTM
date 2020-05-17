import layer
import math
import logging
import numpy as np


class MultilayerNetwork(object):

    def __init__(self, num_input_nodes, num_hidden_layers, num_lstm_layers, num_nodes_per_hidden_layer, num_output_nodes):
        #the number of input nodes are always 4 as the example object has 4 inputs(x).
        self.num_input_nodes = num_input_nodes
        #the number of hidden layers in the model
        self.num_hidden_layers = num_hidden_layers
        #the number of lstm layers in the model
        self.num_lstm_layers = num_lstm_layers
        #the number of nodes for each hidden layer.
        self.num_nodes_per_hidden_layer = num_nodes_per_hidden_layer
        #the number of output nodes are always 1 as the example object has 1 output(y)
        self.num_output_nodes = num_output_nodes

        # create input layer with 0 inputs as it is the initial layer. False attribute is to specify whether this layer is lstm or not
        self.input_layer = layer.Layer(num_input_nodes, 0, False)

        # create hidden layers with the number of inputs as an argument. False attribute is to specify whether this layer is lstm or not
        self.hidden_layers = []
        #the first hidden layer has the number of inputs as the number of nodes in the input layer. 
        self.hidden_layers.append(layer.Layer(num_nodes_per_hidden_layer, num_input_nodes, False))
        for i in range(num_hidden_layers - 1):
            #the hidden layers except the first one will be having the number of inputs as the number of nodes in the previous hidden layer
            self.hidden_layers.append(layer.Layer(num_nodes_per_hidden_layer, num_nodes_per_hidden_layer, False))
        #the hidden layer after the lstm layer will have 1 as the inputs.
        self.hidden_layers[int(num_hidden_layers/2)] = layer.Layer(num_nodes_per_hidden_layer, 1, False)

        # create lstm layers
        self.lstm_layers = []
        #the lstm layer will have the number of nodes in the hidden layer as the number of inputs.
        self.lstm_layers.append(layer.Layer(1, num_nodes_per_hidden_layer, True))

        #self.lstm_layers.append(layer.Layer(num_nodes_per_hidden_layer, num_input_nodes, True))
        for i in range(num_lstm_layers - 1):
            self.lstm_layers.append(layer.Layer(1, 1, True))

        # create output layer
        self.output_layer = layer.Layer(num_output_nodes, num_nodes_per_hidden_layer, False)

    def num_nodes(self):

        return self.num_input_nodes + self.num_hidden_layers * self.num_nodes_per_hidden_layer + self.num_lstm_layers + self.num_output_nodes

    def num_layers(self):

        return 2 + self.num_hidden_layers + self.num_lstm_layers

    def get_layer(self, l):
        #This function determines the specific order of layers in the neural network
        
        if l == 0:
            return self.input_layer
        elif 0 < l <= int(self.num_hidden_layers/2):
            return self.hidden_layers[l - 1]
        elif int(self.num_hidden_layers/2) < l <= int(self.num_hidden_layers/2) + self.num_lstm_layers:
            return self.lstm_layers[l - int(self.num_hidden_layers/2) - 1]
        elif (int(self.num_hidden_layers/2) + self.num_lstm_layers) < l <= ((self.num_hidden_layers) + self.num_lstm_layers):
            return self.hidden_layers[l - (self.num_lstm_layers) - 1]
        elif l == self.num_layers() - 1:
            return self.output_layer
        else:
            return None

    def get_node_in_layer(self, l, n):
        #print(l)
        #print(n)
        return self.get_layer(l).nodes[n]

    def position_in_network(self, l, n):
        pos = n
        for i in range(l):
            pos += self.get_layer(i).num_nodes

        return pos

    def load_weights(self, weights):
        i = 0
        for l in range(1, self.num_layers()):
            for n in range(self.get_layer(l).num_nodes):
                for w in range(len(self.get_node_in_layer(l, n).weights)):
                    self.get_node_in_layer(l, n).weights[w] = weights[i]
                    i += 1

    def weight_string(self, round=False):
        weight_string = '['
        for l in range(1, self.num_layers()):
            for n in range(self.get_layer(l).num_nodes):
                weights = self.get_node_in_layer(l, n).weights
                for w in range(len(weights)):
                    if round:
                        weight_string += ' {0:.3f} '.format(weights[w])
                    else:
                        weight_string += ' {0} '.format(weights[w])
        weight_string += ']'

        return weight_string


def sigmoid(x):

    #print(x)
    if x < 0:
        return 1.0 - 1.0 / (1.0 + math.exp(x))
    else:
        return 1.0/ (1.0 + np.exp(-x))


def sigmoid_derivative(x):

    return sigmoid(x) * (1.0 - sigmoid(x))