%% Initialization
clear ; close all; clc

% read the training data
% first row is column header, first column is label for the training set (actual digit)
digitData = dlmread('train_short100.csv', ',', 1, 0); 

input_layer_size = sqrt(size(digitData)(2)-1);  % input image is 28x28 pixels
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10  ("0" is mapped to label 10)

% initialize neural network parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  Check gradients by running checkNNGradients
checkNNGradients;
