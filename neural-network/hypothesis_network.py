import back_prop_learning
import numpy as np

class HypothesisNetwork(object):

    def __init__(self, network):
        self.network = network

    def guess(self, input):
        #this section takes in the inputs of example object and runs through the trained network 
        #and gives the output in the output layer. 
        
        # load in the input and feed forward from the input layer and through the network
        back_prop_learning.load_and_feed(input, self.network)
        output_layer = self.network.output_layer

        #puts the output in the output layer.
        
        output = []
        for i in range(output_layer.num_nodes):
            output.append(output_layer.nodes[i].output)

        return output
