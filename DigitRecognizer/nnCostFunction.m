function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%%%%%%%%%%%%%%%%%%%%%%
% forward propagation

a1_0 = ones(m, 1);
a1 = [a1_0, X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

temp = zeros(m, num_labels);
for i=1:m
    temp(i, y(i)) = 1;
end

J =  ( -log(a3).*temp - log(1 - a3).*(1 - temp) );
J = sum(sum(J))/m;

%%%%%%%%%%%%%%%%%%%%%%
% regularization of cost function J

temp_theta1 = Theta1;
temp_theta1(:,1) = 0;
temp_theta2 = Theta2;
temp_theta2(:,1) = 0;

J = J + ( sum(sumsq(temp_theta1)) + sum(sumsq(temp_theta2)) )*lambda/(2*m);

%%%%%%%%%%%%%%%%%%%%%%
% back propagation

temp = zeros(m, num_labels);
for i=1:m
    temp(i, y(i)) = 1;
end
delta3 = a3 - temp;
delta2 = (delta3*Theta2).*[ones(m,1), sigmoidGradient(z2)];
delta2 = delta2(:,2:end);
Theta2_grad = Theta2_grad + delta3'*a2/m;
Theta1_grad = Theta1_grad + delta2'*a1/m;

%%%%%%%%%%%%%%%%%%%%%%
% regularization of cost function J

Theta1_grad = Theta1_grad + lambda*temp_theta1/m;
Theta2_grad = Theta2_grad + lambda*temp_theta2/m;


%%%%%%%%%%%%%%%%%%%%%%
% unroll the gradient into a vector for minimization

grad = [Theta1_grad(:); Theta2_grad(:)];


return
