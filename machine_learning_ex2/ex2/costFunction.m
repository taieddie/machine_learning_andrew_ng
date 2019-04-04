function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Basically we calculate the cost for all rows, instead of each row. Then,
% we just take the sum of the values, which is equivalent to taking the sum
% in the logic regression equation.

J = sum(-y.*log(sigmoid(X*theta)) - ((1-y).*log(1-sigmoid(X*theta))), 1) / m;

% Each feature is multiplied by the "cost". Then, we take the sum
% (column-wise) to get the total sum for each feature. The result is
% divided by m, to get the average for each feature.
grad = sum((sigmoid(X*theta) - y) .*X, 1) / m;





% =============================================================

end
