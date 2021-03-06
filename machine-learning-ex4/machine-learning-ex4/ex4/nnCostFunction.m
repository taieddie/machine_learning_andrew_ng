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
y_zeros = zeros(num_labels,1);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Now that the k dimension is here, each yk(i) is a 10 dimensional vector
X = [ones(m, 1) X];
cost = 0;

hidden_layer = sigmoid(X*Theta1');
hidden_layer = [ones(m, 1) hidden_layer];
% For each class, multiply the theta by h(x)
h_x = sigmoid(hidden_layer*Theta2');

for each_feature = 1:m
    digit_index_1 = y(each_feature);
    
    %{ 
    This block of code does the entire multiplication, rather than
    skipping the ones that aren't needed because the y values are zero for
    labels that aren't the predicted value
    temp = y_zeros;
    temp(digit_index_1) = 1;
    
    first_part = -1 .* temp .* log(output(each_feature,:))';
    
    %cost = cost + sum(first_part); 
    
    % Not as easy because it is 1 - y(i)k
    second_part_1 = 1 - temp;
    
    second_part_2 = log(1 - output(each_feature,:))';
    
    cost = cost + sum(first_part) + sum(-1.*second_part_1.*second_part_2);
    %}
    
    % We just need to calculate h(x) for label y(feature)
    % because y is just zero for the other labels
    first_part = -1 * log(h_x(each_feature,digit_index_1));
    
    cost = cost + first_part; 
    
    temp = y_zeros;
    temp(digit_index_1) = 1;
    
    % Not as easy because it is 1 - y(i)k
    second_part_1 = 1 - temp;
    
    second_part_2 = log(1 - h_x(each_feature,:))';
    
    cost = cost + sum(-1.*second_part_1.*second_part_2);
    %}
end

     
J = cost / m;

% Add regularization
% Need to make sure not to include the bias unit, which is the first column
theta_1_squared = Theta1(:,2:end).^2;
theta_1_sum = sum(theta_1_squared(:));
theta_2_squared = Theta2(:,2:end).^2;
theta_2_sum = sum(theta_2_squared(:));
regularization_calculation = (theta_1_sum+theta_2_sum) * (lambda / (2*m));

J = J + regularization_calculation;



% Backpropagation
% a_1 = x
% z_2 = theta_1 * a_1
% a_2 = g(z_2)
% z_3 = theta_2 * a_2
% a_3 = g(z_3) = h(x)

delta_2 = zeros(size(Theta1,1), size(X,2));

for each_feature = 1:m
    digit_index_1 = y(each_feature);
    
    a_1 = X(each_feature,:);
    
    %a_2 = sigmoid(a_1*Theta1');
    %a_2 = [1 a_2];
    
    % Don't recalculate
    a_2 = hidden_layer(each_feature,:);
  
    %a_3 = sigmoid(a_2*Theta2');
    % Don't recalculate
    a_3 = h_x(each_feature,:);
    
    y_k = y_zeros;
    y_k(digit_index_1) = 1;
    
    d_3 = a_3' - y_k;
    
    % Go backwards
    d_2 = (Theta2'*d_3).*(a_2.*(1-a_2))';
    
    Theta1_grad = Theta1_grad + (d_2(2:end) * a_1);
    Theta2_grad = Theta2_grad + (d_3 * a_2);
    
end

Theta1_grad = Theta1_grad ./ m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) .* Theta1(:,2:end);

Theta2_grad = Theta2_grad ./ m;

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) .* Theta2(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
