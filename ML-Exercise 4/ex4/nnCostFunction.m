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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


X = [ones(m,1) X];
size(X);

z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1) a2]; % add the hidden layer bias (a2_0)
z3 = a2 * Theta2';
predictions = sigmoid(z3);

size(predictions);

% map y in a m x num_labels vector (like predictions)
ys = zeros(m,num_labels);
for i = 1:m
    val = y(i);
    if val == 0
        val = 10;
    endif
    ys(i, val) = 1;
endfor;

% check if decoded correctly
%sel = randperm(size(X, 1));
%sel = sel(1:10);
%y(sel, :)
%ys(sel, :)

% inverse transformation
%[vmax, p] = max(all_predictions, [], 2);

% calculate the error for each prediction (size: m x num_labels)
temp = sum(ys .* log(predictions) + (1 .- ys) .* log(1 .- predictions), 2);
J = -1/m * sum(temp);

% regularization of the cost function
Theta1_cols = size(Theta1, 2);
Theta2_cols = size(Theta2, 2);
J = J + lambda / (2*m) * ( sum(sum(Theta1(:, 2:Theta1_cols).^2, 2)) + sum(sum(Theta2(:,2:Theta2_cols).^2, 2)) );



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

for t = 1:m
    a_1 = X(t, :)';
    
    z_2 = Theta1 * a_1; 
    a_2 = sigmoid(z_2); % hidden layer output
    a_2 = [1; a_2]; % add the hidden layer bias (a2_0)
    
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3); % output layer output
    
    delta_3 = a_3 .- ys(t,:)';
    
    % Need to add a 1 on z_2 for the bias unit of the hidden layer
    delta_2 = Theta2'*delta_3 .* [1; sigmoidGradient(z_2)];
    delta_2 = delta_2(2:end);
    % Equally correct:
    %delta_2 = Theta2'*delta_3 ;
    %delta_2 = delta_2(2:end).* sigmoidGradient(z_2);
    Delta_1 = Delta_1 .+ delta_2 * a_1';
    Delta_2 = Delta_2 .+ delta_3 * a_2';
endfor;

Theta1_grad = Delta_1 ./m;
Theta2_grad = Delta_2 ./m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_reg = lambda/m * Theta1;
Theta1_reg(:,1) = zeros(size(Theta1_reg,1),1); % no regulation for the bias unit
Theta2_reg = lambda/m * Theta2;
Theta2_reg(:,1) = zeros(size(Theta2_reg,1),1); % no regulation for the bias unit

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
