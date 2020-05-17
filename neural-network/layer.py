import random
import numpy as np
import perceptron


class Layer(object):

    def __init__(self, num_nodes, num_inputs_per_node, is_lstm):
        self.num_nodes = num_nodes
        self.is_lstm = is_lstm
        #is_lstm is to identify if the layer is a lstm layer or not. 
        
        
        #stacking x(present input xt) and h(t-1)
		# xc = np.hstack((x,  h_prev))
		# #dot product of Wf(forget weight matrix and xc +bias)
		# self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
		# #finally multiplying forget_gate(self.state.f) with previous cell state(s_prev) to get present state.
		# self.state.s = self.state.g * self.state.i + s_prev * self.state.f

        # this section is for creating perceptron/nodes. For every node, the number of inputs is passed into the perceptron object
        self.nodes = []
        for i in range(num_nodes):
            self.nodes.append(perceptron.Perceptron(num_inputs_per_node, is_lstm))