function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% Cost = 1/2m * Sum of (h(theta)(xi) - yi)^2
% h(theta)(xi) = X (matrix) * theta

% The square of the values are done element-wise
J = (1/(2*m)) * sum((X*theta - y).^2);

% =========================================================================

end
