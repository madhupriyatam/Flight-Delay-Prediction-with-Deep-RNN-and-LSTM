import random
import hypothesis_network
import multilayer_network
import logging
import numpy as np


def back_prop_learning(examples, network, alpha=0.1, iteration_max=5000000, weights=None, verbose=False):

    delta = [0] * network.num_nodes()
    # a vector of errors, indexed by network node
    delta_Wxi = [0] * network.num_nodes()
    delta_Whf = [0] * network.num_nodes()
    delta_Wht = [0] * network.num_nodes()
    delta_Wci = [0] * network.num_nodes()
    delta_Wxf = [0] * network.num_nodes()
    delta_Wcf = [0] * network.num_nodes()
    delta_Wxc = [0] * network.num_nodes()
    delta_Whc = [0] * network.num_nodes()
    delta_Wxo = [0] * network.num_nodes()
    delta_Who = [0] * network.num_nodes()
    delta_Wco = [0] * network.num_nodes()

    # keep learning until stopping criterion is satisfied
    for iteration in range(iteration_max):
        #we have set the iterations as 1000
        print("Iteration No. ",iteration)
        #this formula is for the decay of the alpha factor
        new_alpha = alpha * (1 - (float(iteration) / iteration_max))
        #print("Alpha is: " + str(alpha))
        learn_loop(delta,delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco, examples, network, new_alpha)
        if verbose:
            logging.info('Neural network learning loop {0} of {1} with alpha: {2}'.format(iteration, iteration_max,
                                                                                          new_alpha))

    return hypothesis_network.HypothesisNetwork(network)


def learn_loop(delta,delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco,examples, network, alpha):

    for example in examples:
        load_and_feed(example.x, network)

        # compute the Mean squared error at the output
        for n in range(network.output_layer.num_nodes):
            #print("Output: ",network.output_layer.nodes[n].output)
            #loading the gradient of output layer initially
            delta[network.position_in_network(network.num_layers() - 1, n)] = \
                multilayer_network.sigmoid_derivative(network.output_layer.nodes[n].in_sum) * \
                (((example.y[n] - network.output_layer.nodes[n].output) ** 2) / 2)

        # back propagating through time: the gradients backward from output layer to input layer
        delta_propagation(delta,delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco,network)

        # update every weight in the network using gradients
        update_weights(delta, delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco,network, alpha)


def load_and_feed(input, network):

    #the nodes in the input layer are loaded with example objects value
    for i in range(len(network.input_layer.nodes)):
        network.input_layer.nodes[i].output = input[i]

    # feeding the inputs forward from input layer to output layer: computing the outputs
    feed_forward(network)


def feed_forward(network):

    for l in range(1, network.num_layers()):
        for n in range(network.get_layer(l).num_nodes):
            node = network.get_node_in_layer(l, n)

            if network.get_layer(l).is_lstm:
                if not network.get_layer(l-1).is_lstm:
                    #this layer is lstm layer and the previous layer is a hidden layer
                    
                    in_sum_Wxi = 0.0
                    in_sum_Wht = 0.0
                    in_sum_Wci = 0.0

                    in_sum_Wxf = 0.0
                    in_sum_Whf = 0.0
                    in_sum_Wcf = 0.0

                    in_sum_Wxc = 0.0
                    in_sum_Whc = 0.0

                    in_sum_Wxo = 0.0
                    in_sum_Who = 0.0
                    in_sum_Wco = 0.0
                    #finding the insum values for each input for a particular node
                    for i in range(node.num_inputs):
                        #implementation of Wxt
                        in_sum_Wxi += node.Wxi[i] * network.get_node_in_layer(l-1, i).output
                        #print("node.Wxi[i] = " + str(node.Wxi))
                        #print("network.get_node_in_layer(l-1, i).output = " + str(network.get_node_in_layer(l-1, i).output))
                        #implementation of Wh(t-1)
                        in_sum_Wht += node.Wht[i] * node.ht
                        #implementation of Wc(t-1)
                        in_sum_Wci += node.Wci[i] * node.ct

                        in_sum_Wxf += node.Wxf[i] * network.get_node_in_layer(l-1, i).output
                        in_sum_Whf += node.Whf[i] * node.ht
                        in_sum_Wcf += node.Wcf[i] * node.ct

                        in_sum_Wxc += node.Wxc[i] * network.get_node_in_layer(l-1, i).output
                        in_sum_Whc += node.Whc[i] * node.ht

                        in_sum_Wxo += node.Wxo[i] * network.get_node_in_layer(l-1, i).output
                        in_sum_Who += node.Who[i] * node.ht
                        in_sum_Wco += node.Wco[i] * node.ct
                    #implementation of W * x + bias
                    sum_i = in_sum_Wxi + in_sum_Wht + in_sum_Wci + node.Wxi[len(node.Wxi) - 1]+ node.Wht[len(node.Wht) - 1]+ node.Wci[len(node.Wci) - 1]#node.bi
                    sum_f = in_sum_Wxf + in_sum_Whf + in_sum_Wcf + node.Wxf[len(node.Wxf) - 1]+ node.Whf[len(node.Whf) - 1]+ node.Wcf[len(node.Wcf) - 1]#node.bf
                    sum_c = in_sum_Wxc + in_sum_Whc + node.Wxc[len(node.Wxc) - 1]+ node.Whc[len(node.Whc) - 1]#node.bc
                    sum_o = in_sum_Wxo + in_sum_Who + in_sum_Wco + node.Wxo[len(node.Wxo) - 1]+ node.Who[len(node.Who) - 1]+ node.Wco[len(node.Wco) - 1]#node.bo
                    
                    #IMPLEMENTATION OF THE LSTM CELL GATES USING THE FORMULA MENTIONED IN THE RESEARCH PAPER
                    
                    #it = φ (Wxixt + Whtht−1 + Wcict−1 + bi)
                    #ft = φ (Wxfxt + Whfht−1 + Wcf ct−1 + bf )
                    #ct = ftct−1 + it tanh (Wxcxt + Whcht−1 + bc)
                    #ot = φ (Wxoxt + Whoht−1 + Wcoct + bo)
                    #ht = ot tanh (ct)
                    
                    #print("sum_i = " + str(sum_i))
                    #print("sum_f = " + str(sum_f))
                    #print("sum_c = " + str(sum_c))
                    #print("sum_o = " + str(sum_o))

                    #network.get_node_in_layer(l, n).in_sum = summation
		            #network.get_node_in_layer(l, n).output = multilayer_network.sigmoid(summation)

		            #network.get_node_in_layer(l, n).sum_ct = sum_c
                    network.get_node_in_layer(l, n).ct = multilayer_network.sigmoid(sum_f) * network.get_node_in_layer(l, n).ct + \
		    									multilayer_network.sigmoid(sum_i) * np.tanh(sum_c)
                    #print("in_sum_Wxo",in_sum_Wxo)
                    #print("in_sum_Who", in_sum_Who)
                    #print("in_sum_Wco", in_sum_Wco)
                    #print("Sum_O:",sum_o)

                    #print("ct = " + str(network.get_node_in_layer(l, n).ct))
                    sig_ot = multilayer_network.sigmoid(sum_o)


                    #network.get_node_in_layer(l, n).in_sum = summation
                    network.get_node_in_layer(l, n).ht = sig_ot * np.tanh(network.get_node_in_layer(l, n).ct)
                    #STORING the insum values for using it in the back propagation.
                    network.get_node_in_layer(l, n).in_sum_Wxi = in_sum_Wxi
                    network.get_node_in_layer(l, n).in_sum_Wht = in_sum_Wht
                    network.get_node_in_layer(l, n).in_sum_Wci = in_sum_Wci
                    network.get_node_in_layer(l, n).in_sum_Wxf = in_sum_Wxf
                    network.get_node_in_layer(l, n).in_sum_Whf = in_sum_Whf
                    network.get_node_in_layer(l, n).in_sum_Wcf = in_sum_Wcf
                    network.get_node_in_layer(l, n).in_sum_Wxc = in_sum_Wxc
                    network.get_node_in_layer(l, n).in_sum_Whc = in_sum_Whc
                    network.get_node_in_layer(l, n).in_sum_Wxo = in_sum_Wxo
                    network.get_node_in_layer(l, n).in_sum_Who = in_sum_Who
                    network.get_node_in_layer(l, n).in_sum_Wco = in_sum_Wco

                    #print("ht = " + str(network.get_node_in_layer(l, n).ht))
                    


                #if l-1 is hidden
            			#compute taking output values of l-1 layer as x(t) and h(t) using old h(t) value

                else:
                    # compute taking h(t) from l-1 layer as x(t)
                    #this layer is lstm and the previous layer is lstm layer 
                    in_sum_Wxi = 0.0
                    in_sum_Wht = 0.0
                    in_sum_Wci = 0.0

                    in_sum_Wxf = 0.0
                    in_sum_Whf = 0.0
                    in_sum_Wcf = 0.0

                    in_sum_Wxc = 0.0
                    in_sum_Whc = 0.0

                    in_sum_Wxo = 0.0
                    in_sum_Who = 0.0
                    in_sum_Wco = 0.0
                    #finding the insum values for each input for a particular node
                    for i in range(node.num_inputs):
                        #implementation of Wxt
                        in_sum_Wxi += node.Wxi[i] * network.get_node_in_layer(l-1, i).ht
                        #implementation of Wh(t-1)
                        in_sum_Wht += node.Wht[i] * node.ht
                        #implementation of Wc(t-1)
                        in_sum_Wci += node.Wci[i] * node.ct

                        in_sum_Wxf += node.Wxf[i] * network.get_node_in_layer(l-1, i).ht
                        in_sum_Whf += node.Whf[i] * node.ht
                        in_sum_Wcf += node.Wcf[i] * node.ct

                        in_sum_Wxc += node.Wxc[i] * network.get_node_in_layer(l-1, i).ht
                        in_sum_Whc += node.Whc[i] * node.ht

                        in_sum_Wxo += node.Wxo[i] * network.get_node_in_layer(l-1, i).ht
                        in_sum_Who += node.Who[i] * node.ht
                        in_sum_Wco += node.Wco[i] * node.ct
                    #implementation of W * x + bias
                    sum_i = in_sum_Wxi + in_sum_Wht + in_sum_Wci + node.Wxi[len(node.Wxi) - 1]+ node.Wht[len(node.Wht) - 1]+ node.Wci[len(node.Wci) - 1]#node.bi
                    sum_f = in_sum_Wxf + in_sum_Whf + in_sum_Wcf + node.Wxf[len(node.Wxf) - 1]+ node.Whf[len(node.Whf) - 1]+ node.Wcf[len(node.Wcf) - 1]#node.bf
                    sum_c = in_sum_Wxc + in_sum_Whc + node.Wxc[len(node.Wxc) - 1]+ node.Whc[len(node.Whc) - 1]#node.bc
                    sum_o = in_sum_Wxo + in_sum_Who + in_sum_Wco + node.Wxo[len(node.Wxo) - 1]+ node.Who[len(node.Who) - 1]+ node.Wco[len(node.Wco) - 1]#node.bo

                    #IMPLEMENTATION OF THE LSTM CELL GATES USING THE FORMULA MENTIONED IN THE PAPER
                    
                    #it = φ (Wxixt + Whtht−1 + Wcict−1 + bi)
                    #ft = φ (Wxfxt + Whfht−1 + Wcf ct−1 + bf )
                    #ct = ftct−1 + it tanh (Wxcxt + Whcht−1 + bc)
                    #ot = φ (Wxoxt + Whoht−1 + Wcoct + bo)
                    #ht = ot tanh (ct)
     
                    #network.get_node_in_layer(l, n).in_sum = summation
                    #network.get_node_in_layer(l, n).output = multilayer_network.sigmoid(summation)

                    #network.get_node_in_layer(l, n).sum_ct = sum_c
                    #storing the Ct value for next time interval
                    network.get_node_in_layer(l, n).ct = multilayer_network.sigmoid(sum_f) * network.get_node_in_layer(l, n).ct + \
                                                multilayer_network.sigmoid(sum_i) * np.tanh(sum_c)

                    sig_ot = multilayer_network.sigmoid(sum_o) 

                    #network.get_node_in_layer(l, n).in_sum = summation
                    #storing the Ht for the next time interval
                    network.get_node_in_layer(l, n).ht = sig_ot * np.tanh(network.get_node_in_layer(l, n).ct)
                    #storing the in_sum values for using them in back propagation
                    network.get_node_in_layer(l, n).in_sum_Wxi = in_sum_Wxi
                    network.get_node_in_layer(l, n).in_sum_Wht = in_sum_Wht
                    network.get_node_in_layer(l, n).in_sum_Wci = in_sum_Wci
                    network.get_node_in_layer(l, n).in_sum_Wxf = in_sum_Wxf
                    network.get_node_in_layer(l, n).in_sum_Whf = in_sum_Whf
                    network.get_node_in_layer(l, n).in_sum_Wcf = in_sum_Wcf
                    network.get_node_in_layer(l, n).in_sum_Wxc = in_sum_Wxc
                    network.get_node_in_layer(l, n).in_sum_Whc = in_sum_Whc
                    network.get_node_in_layer(l, n).in_sum_Wxo = in_sum_Wxo
                    network.get_node_in_layer(l, n).in_sum_Who = in_sum_Who
                    network.get_node_in_layer(l, n).in_sum_Wco = in_sum_Wco


                #print("LSTM layer: " + str(l) + ", node:" + str(n) + "output is:" + str(network.get_node_in_layer(l,n).ht))

            else:
                if network.get_layer(l-1).is_lstm:
                    #this layer is hidden layer and the previous layer is a lstm layer
                    summation = 0.0
                    #Implementation of W * x + b
                    #Ht is taken into consideration as the previous layer is lstm
                    for i in range(node.num_inputs):
                        summation += node.weights[i] * network.get_node_in_layer(l - 1, i).ht
                    summation += node.weights[len(node.weights) - 1]    # bias input

                    # network.get_node_in_layer(l, n).in_sum = summation
                    # network.get_node_in_layer(l, n).output = multilayer_network.sigmoid(summation)

                else:
                    #this layer is hidden layer and the previous layer is a hidden layer
                    summation = 0.0
                    #implementation of W * x + b
                    #output is taken into consideration as the previous layer is a hidden layer
                    for i in range(node.num_inputs):
                        summation += node.weights[i] * network.get_node_in_layer(l - 1, i).output
                    summation += node.weights[len(node.weights) - 1]    # bias input

                network.get_node_in_layer(l, n).in_sum = summation
                network.get_node_in_layer(l, n).output = multilayer_network.sigmoid(summation)

                #print("hidden layer: " + str(l) + ", node:" + str(n) + "output is:" + str(network.get_node_in_layer(l,n).output))

					

            #network.get_node_in_layer(l, n).in_sum = summation
		    #network.get_node_in_layer(l, n).output = network.get_node_in_layer(l, n).ot * np.tanh(network.get_node_in_layer(l, n).ct)

            #compute ht here for lstm layers by using recuurent formulas given in paper
            #calculate ct here by using ct-1 stored in the same way as ht-1 in perceptron


def delta_propagation(delta, delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco,network):

    for l in range(network.num_layers() - 2, 0, -1):
        for n in range(network.get_layer(l).num_nodes):

            if network.get_layer(l).is_lstm:
                if not network.get_layer(l+1).is_lstm:
                    #this layer is lstm and above layer is a hidden layer. Here, the above layer's weights and delta are considered as 
                    #summation. This summation is used for lstm weights delta.
                    summation = 0.0
                    next_layer_nodes = network.get_layer(l + 1).nodes
                    for nln in range(len(next_layer_nodes)):
                        summation += next_layer_nodes[nln].weights[n] * delta[network.position_in_network(l + 1, nln)]

                    # "blame" a node as much as its weight
                    #delta[network.position_in_network(l, n)] = \
                    #    multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation

                    delta_Wht[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wht) * summation

                    delta_Wxi[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxi) * summation

                    delta_Wci[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wci) * summation

                    delta_Wxf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxf) * summation
                    
                    delta_Whf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Whf) * summation

                    delta_Wcf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wcf) * summation

                    delta_Wxc[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxc) * summation

                    delta_Whc[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Whc) * summation

                    delta_Wxo[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxo) * summation

                    delta_Who[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Who) * summation

                    delta_Wco[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wco) * summation

                else:
                    summation = 0.0
                    summation_Wxi = 0.0
                    summation_Wht = 0.0
                    summation_Wci = 0.0
                    summation_Wxf = 0.0
                    summation_Whf = 0.0
                    summation_Wcf = 0.0
                    summation_Wxc = 0.0
                    summation_Whc = 0.0
                    summation_Wxo = 0.0
                    summation_Who = 0.0
                    summation_Wco = 0.0

                    #this layer is a lstm layer and the above layer is a lstm layer. Here, the above layer's weights and delta's are considered as 
                    #summation. This summation is used for lstm weights delta.
                    next_layer_nodes = network.get_layer(l + 1).nodes
                    for nln in range(len(next_layer_nodes)):
                        summation_Wxi += next_layer_nodes[nln].Wxi[n] * delta_Wxi[network.position_in_network(l + 1, nln)]
                        summation_Wht += next_layer_nodes[nln].Wht[n] * delta_Wht[network.position_in_network(l + 1, nln)]
                        summation_Wci += next_layer_nodes[nln].Wci[n] * delta_Wci[network.position_in_network(l + 1, nln)]
                        summation_Wxf += next_layer_nodes[nln].Wxf[n] * delta_Wxf[network.position_in_network(l + 1, nln)]
                        summation_Whf += next_layer_nodes[nln].Whf[n] * delta_Whf[network.position_in_network(l + 1, nln)]
                        summation_Wcf += next_layer_nodes[nln].Wcf[n] * delta_Wcf[network.position_in_network(l + 1, nln)]
                        summation_Wxc += next_layer_nodes[nln].Wxc[n] * delta_Wxc[network.position_in_network(l + 1, nln)]
                        summation_Whc += next_layer_nodes[nln].Whc[n] * delta_Whc[network.position_in_network(l + 1, nln)]
                        summation_Wxo += next_layer_nodes[nln].Wxo[n] * delta_Wxo[network.position_in_network(l + 1, nln)]
                        summation_Who += next_layer_nodes[nln].Who[n] * delta_Who[network.position_in_network(l + 1, nln)]
                        summation_Wco += next_layer_nodes[nln].Wco[n] * delta_Wco[network.position_in_network(l + 1, nln)]


            # "blame" a node as much as its weight
                    #delta[network.position_in_network(l, n)] = \
                    #    multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation

                    delta_Wxi[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxi) * summation_Wxi

                    delta_Wht[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wht) * summation_Wht

                    delta_Wci[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wci) * summation_Wci

                    delta_Wxf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxf) * summation_Wxf
                    
                    delta_Whf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Whf) * summation_Whf

                    delta_Wcf[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wcf) * summation_Wcf

                    delta_Wxc[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxc) * summation_Wxc

                    delta_Whc[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Whc) * summation_Whc

                    delta_Wxo[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wxo) * summation_Wxo

                    delta_Who[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Who) * summation_Who

                    delta_Wco[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum_Wco) * summation_Wco

            else:
                if network.get_layer(l+1).is_lstm:
                    summation = 0.0
                    next_layer_nodes = network.get_layer(l + 1).nodes
                    #this layer is a hidden layer and the above layer is a lstm layer. The delta of the lstm weights are combined 
                    #to form a summation. 
                    for nln in range(len(next_layer_nodes)):
                        summation += \
                        (next_layer_nodes[nln].Wxi[n] * delta_Wxi[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wht[n] * delta_Wht[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wci[n] * delta_Wci[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wxf[n] * delta_Wxf[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Whf[n] * delta_Whf[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wcf[n] * delta_Wcf[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wxc[n] * delta_Wxc[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Whc[n] * delta_Whc[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wxo[n] * delta_Wxo[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Who[n] * delta_Who[network.position_in_network(l + 1, nln)]) + \
                        (next_layer_nodes[nln].Wco[n] * delta_Wco[network.position_in_network(l + 1, nln)])

                    # "blame" a node as much as its weight
                    delta[network.position_in_network(l, n)] = \
                        multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation

                else:
                    #this layer is a hidden layer and the above layer is a hidden layer too. 
                    #the backpropagation is done by the summation of multiplying the weights of the hidden layer nodes with that of respecitve deltas.  
                    summation = 0.0
                    next_layer_nodes = network.get_layer(l + 1).nodes
                    for nln in range(len(next_layer_nodes)):
                        summation += next_layer_nodes[nln].weights[n] * delta[network.position_in_network(l + 1, nln)]

                    # "blame" a node as much as its weight
                    delta[network.position_in_network(l, n)] = \
                       multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation

            # # "blame" a node as much as its weight
            #         delta[network.position_in_network(l, n)] = \
            #             multilayer_network.sigmoid_derivative(network.get_node_in_layer(l, n).in_sum) * summation



def update_weights(delta,delta_Wxi, delta_Whf,delta_Wht,delta_Wci,delta_Wxf,delta_Wcf,delta_Wxc,delta_Whc,delta_Wxo,delta_Who,delta_Wco, network, alpha):

    for l in range(1, network.num_layers()):
        for n in range(network.get_layer(l).num_nodes):
            # adjust the weights
            #In this function, after all the deltas are loaded, the weights are updated with their respective gradient list values 
            if network.get_layer(l).is_lstm:
                if not network.get_layer(l-1).is_lstm:
                    node = network.get_node_in_layer(l, n)
                    #This is a Lstm layer and the previous layer is hidden. The Learning rate is multiplied with the output of the previous and the respective delta values. 
                    
                    for i in range(node.num_inputs):
                        node.Wxi[i] -= alpha * network.get_node_in_layer(l - 1, i).output * \
                                           delta_Wxi[network.position_in_network(l, n)]

                        node.Wxf[i] -= alpha * network.get_node_in_layer(l - 1, i).output * \
                                           delta_Wxf[network.position_in_network(l, n)]

                        node.Wxc[i] -= alpha * network.get_node_in_layer(l - 1, i).output * \
                                           delta_Wxc[network.position_in_network(l, n)]

                        node.Wxo[i] -= alpha * network.get_node_in_layer(l - 1, i).output * \
                                           delta_Wxo[network.position_in_network(l, n)]

                        node.Wht[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Wht) * \
                                           delta_Wht[network.position_in_network(l, n)]

                        node.Wci[i] += alpha * multilayer_network.sigmoid(node.in_sum_Wci) * \
                                           delta_Wci[network.position_in_network(l, n)]

                        node.Whf[i] += alpha * multilayer_network.sigmoid(node.in_sum_Whf) * \
                                           delta_Whf[network.position_in_network(l, n)]

                        node.Wcf[i] += alpha * multilayer_network.sigmoid(node.in_sum_Wcf) * \
                                           delta_Wcf[network.position_in_network(l, n)]

                        node.Whc[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Whc) * \
                                           delta_Whc[network.position_in_network(l, n)]

                        node.Who[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Who) * \
                                           delta_Who[network.position_in_network(l, n)]

                        node.Wco[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Wco) * \
                                           delta_Wco[network.position_in_network(l, n)]
                    #This updation is for the bias of the all the weights. 
                    node.Wxi[len(node.Wxi) - 1] += alpha * delta_Wxi[network.position_in_network(l, n)]   # bias input
                    node.Wxf[len(node.Wxf) - 1] += alpha * delta_Wxf[network.position_in_network(l, n)]   # bias input
                    node.Wxc[len(node.Wxc) - 1] += alpha * delta_Wxc[network.position_in_network(l, n)]   # bias input
                    node.Wxo[len(node.Wxo) - 1] += alpha * delta_Wxo[network.position_in_network(l, n)]   # bias input
                    node.Wht[len(node.Wht) - 1] += alpha * delta_Wht[network.position_in_network(l, n)]   # bias input
                    node.Wci[len(node.Wci) - 1] += alpha * delta_Wci[network.position_in_network(l, n)]   # bias input
                    node.Whf[len(node.Whf) - 1] += alpha * delta_Whf[network.position_in_network(l, n)]   # bias input
                    node.Wcf[len(node.Wcf) - 1] += alpha * delta_Wcf[network.position_in_network(l, n)]   # bias input
                    node.Whc[len(node.Whc) - 1] += alpha * delta_Whc[network.position_in_network(l, n)]   # bias input
                    node.Who[len(node.Who) - 1] += alpha * delta_Who[network.position_in_network(l, n)]   # bias input
                    node.Wco[len(node.Wco) - 1] += alpha * delta_Wco[network.position_in_network(l, n)]   # bias input

                else:
                    node = network.get_node_in_layer(l, n)
                    #this layer is a lstm layer and and the previous layer is a lstm. So, the ht of the previous layer is taken into consideration. 
                    #
                    for i in range(node.num_inputs):
                        node.Wxi[i] -= alpha * network.get_node_in_layer(l - 1, i).ht * \
                                           delta_Wxi[network.position_in_network(l, n)]

                        node.Wxf[i] -= alpha * network.get_node_in_layer(l - 1, i).ht * \
                                           delta_Wxf[network.position_in_network(l, n)]

                        node.Wxc[i] -= alpha * network.get_node_in_layer(l - 1, i).ht * \
                                           delta_Wxc[network.position_in_network(l, n)]

                        node.Wxo[i] -= alpha * network.get_node_in_layer(l - 1, i).ht * \
                                           delta_Wxo[network.position_in_network(l, n)]

                        node.Wht[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Wht) * \
                                           delta_Wht[network.position_in_network(l, n)]

                        node.Wci[i] += alpha * multilayer_network.sigmoid(node.in_sum_Wci) * \
                                           delta_Wci[network.position_in_network(l, n)]

                        node.Whf[i] += alpha * multilayer_network.sigmoid(node.in_sum_Whf) * \
                                           delta_Whf[network.position_in_network(l, n)]

                        node.Wcf[i] += alpha * multilayer_network.sigmoid(node.in_sum_Wcf) * \
                                           delta_Wcf[network.position_in_network(l, n)]

                        node.Whc[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Whc) * \
                                           delta_Whc[network.position_in_network(l, n)]

                        node.Who[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Who) * \
                                           delta_Who[network.position_in_network(l, n)]

                        node.Wco[i] -= alpha * multilayer_network.sigmoid(node.in_sum_Wco) * \
                                           delta_Wco[network.position_in_network(l, n)]

                    node.Wxi[len(node.Wxi) - 1] += alpha * delta_Wxi[network.position_in_network(l, n)]   # bias input
                    node.Wxf[len(node.Wxf) - 1] += alpha * delta_Wxf[network.position_in_network(l, n)]   # bias input
                    node.Wxc[len(node.Wxc) - 1] += alpha * delta_Wxc[network.position_in_network(l, n)]   # bias input
                    node.Wxo[len(node.Wxo) - 1] += alpha * delta_Wxo[network.position_in_network(l, n)]   # bias input
                    node.Wht[len(node.Wht) - 1] += alpha * delta_Wht[network.position_in_network(l, n)]   # bias input
                    node.Wci[len(node.Wci) - 1] += alpha * delta_Wci[network.position_in_network(l, n)]   # bias input
                    node.Whf[len(node.Whf) - 1] += alpha * delta_Whf[network.position_in_network(l, n)]   # bias input
                    node.Wcf[len(node.Wcf) - 1] += alpha * delta_Wcf[network.position_in_network(l, n)]   # bias input
                    node.Whc[len(node.Whc) - 1] += alpha * delta_Whc[network.position_in_network(l, n)]   # bias input
                    node.Who[len(node.Who) - 1] += alpha * delta_Who[network.position_in_network(l, n)]   # bias input
                    node.Wco[len(node.Wco) - 1] += alpha * delta_Wco[network.position_in_network(l, n)]   # bias input


            else:
                if network.get_layer(l-1).is_lstm:
                    #this is a hidden layer and the previous layer is a lstm. The Ht obtained in the previous layer and the delta value of every node in that layer 
                    #is taken for updating the weights. 
                    node = network.get_node_in_layer(l, n)
                    for i in range(node.num_inputs):
                        node.weights[i] -= alpha * network.get_node_in_layer(l - 1, i).ht * \
                                           delta[network.position_in_network(l, n)]

                    node.weights[len(node.weights) - 1] += alpha * delta[network.position_in_network(l, n)]   # bias input


                else:
                    node = network.get_node_in_layer(l, n)
                    #this is the a hidden layer and the previous layer is a hidden layer. The output obtained in the previous layer and the delta values of every node 
                    #in that layer is taken for updating the weights. 
                    for i in range(node.num_inputs):
                        node.weights[i] -= alpha * network.get_node_in_layer(l - 1, i).output * \
                                           delta[network.position_in_network(l, n)]

                    node.weights[len(node.weights) - 1] += alpha * delta[network.position_in_network(l, n)]   # bias input


