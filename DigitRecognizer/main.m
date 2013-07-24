%% Initialization
clear ; close all; clc
%TODO select training and xv set after random shuffle of data
%TODO implement PCA

% input parameters
hidden_layer_size = 100; 
maxIter    = 200;
lambda_i   = 0.01;
lambda_f   = 10;
num_labels = 10;          % 10 labels, from 1 to 10  ("0" is mapped to label 10)
nCrop      = 4;		  % number of pixels to crop from each side
lambdaList = [0.25, 0.5, 0.75, 1, 2, 3, 4];

% read the training data
% first row is column header, first column is label for the training set (actual digit)
%trainData = dlmread('TrainData.csv', ',', 1, 0); 
trainData = dlmread('train_short1000.csv', ',', 1, 0); 

% crop the training data
if nCrop > 0
    X = cropData(trainData(:,2:end), nCrop);
else
    X = trainData(:,2:end);
end

input_layer_size = size(X)(2);
%X = trainData(:,2:end);
y = trainData(:,1);

% map 0 to 10 in the labels
y( find(y==0) ) = 10;


% initialize neural network parameters
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
size(initial_Theta1)
size(initial_Theta2)

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  Check gradients by running checkNNGradients
% should be removed for actual run
%checkNNGradients(0); % lambda=0
%checkNNGradients(3); % lambda=3

% train the neural network
options = optimset('MaxIter', maxIter);
accuracy = 0;

%for lambda=lambda_i:lambda_f
lambda = lambda_i;
for i=1:length(lambdaList)
    lambda = lambdaList(i);
    costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    tic
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    toc

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

    % prediction on training set

    pred = predict(Theta1, Theta2, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

    %prediction on cross validation data

    xValidData = dlmread('CrossValidationData.csv', ',', 1, 0); 
    if nCrop > 0
	xvX = cropData(xValidData(:,2:end), nCrop);
    else
	xvX = xValidData(:,2:end);
    end
    xvy = xValidData(:,1);
    xValidPrediction = predict(Theta1, Theta2, xvX);
    xValidPrediction( find(xValidPrediction==10) ) = 0;
    newAccuracy = mean(double(xValidPrediction == xvy) * 100);

    fprintf('lambda=%f, Cross Validation Accuracy = %f\n', lambda, newAccuracy);
    if newAccuracy > accuracy
	accuracy = newAccuracy;
	optiTheta1 = Theta1;
	optiTheta2 = Theta2;
	bestLambda = lambda
    end

end
fprintf('\n\n optimum lambda = %d\n best accuracy = %f\n', bestLambda, accuracy);

% prediction on the test data
testData = dlmread('test.csv', ',', 1, 0); 
if nCrop > 0
    testX = cropData(testData(:,1:end), nCrop);
else
    testX = testData(:,1:end);
end
%testX = testData(:,1:end);
tic
testPrediction = predict(optiTheta1, optiTheta2, testX);
testPrediction( find(testPrediction==10) ) = 0;
toc
fileName = sprintf('DigitPredictions_ncrop%d_layer%d_lambda%4.2f.csv', nCrop, hidden_layer_size, bestLambda);
fid = fopen(fileName, 'w')
for i=1:size(testPrediction)
    fprintf(fid, '%d,%d\n', i, testPrediction(i));
end
fclose(fid)
