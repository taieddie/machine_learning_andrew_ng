function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
%Theta1 = [ones(1,size(X, 2)); Theta1];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Each column is a "label" and each row is a sample (x0,x1,x2...)
% Each label of thetas multiplies each sample row.
hidden_layer = sigmoid(X*Theta1');

% hidden_layer has rows for each row sample and columns for each class
% row i and column j represents the h(x) value for a given class for a
% specific row of samples.
% [ h(x)0  h(x)1  h(x)2  . . .] 
% [ h(x)10 h(x)11 h(x)12 . . .] 
% h(x)0 is the hypothesis, theta*x, x =
% first row sample in X, for label 1. 
% h(x)1 is the same, except it is for label 2.
% h(x)10 is the hypothesis, theta*x, 
% x = second row sample in X, for label 1.
hidden_layer = [ones(m, 1) hidden_layer];

% For each class, multiply the theta by h(x)
output = sigmoid(hidden_layer*Theta2');

[max_vals, indices] = max(output,[],2);

p = indices;
% =========================================================================


end
