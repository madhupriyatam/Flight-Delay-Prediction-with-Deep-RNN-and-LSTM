# Flight Delay Prediction with Deep RNN and LSTM architecture

This project is implemented as a part of the Course CSE 571 Artificial Intelligence. 

Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@madhupriyatam 
madhupriyatam
/
Flight-Delay-Prediction-with-Deep-RNN-and-LSTM
1
00
 Code
 Issues 0
 Pull requests 0 Actions
 Projects 0
 Wiki
 Security 0
 Insights
 Settings
Flight-Delay-Prediction-with-Deep-RNN-and-LSTM/README.txt
@madhupriyatam madhupriyatam Add files via upload
2ccf519 2 days ago
48 lines (38 sloc)  6.99 KB
  
﻿Group 18: Artificial Intelligence: CSE 571

----------------------------------------------------------------------------README----------------------------------------------------------------------------------------------------------


This project is a implementation of RNN & LSTM on Flight prediction systems as proposed in the research paper (https://ieeexplore.ieee.org/abstract/document/7778092)

Github link which has the implementation of Flight Prediction systems on Neural Networks: https://github.com/yanxiaoliang/Flight-Delay-Prediction-Based-on-Neural-Networks/blob/master/project-programs-and-data/neural-network
Used the data preprocessing and neural network header file for the implementation as specified in the proposal. We implemented RNN with LSTM cells.

Data used in this project is from Bureau of transports statistics (http://www.transtats.bts.gov.)

Our work includes implementation of RNN architecture via LSTM cells, backpropagating through time, using stochastic 
Gradient Descent(SGD) as the optimizer and training the model with suitable randomized weight techniques. (From Scratch. Without using tensorflow or keras)

--------------------------------------Dependencies-------------------------------------------------------------------
Numpy, python 3.7.0

--------------------------------------STEPS TO RUN THE PROJECT-------------------------------------------------------
1. Unzip the zip file. 
2. Make the neural_network folder as the working directory.  
3. Run test.py file : python test.py  
