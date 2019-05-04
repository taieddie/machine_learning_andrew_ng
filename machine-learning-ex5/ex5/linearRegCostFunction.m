function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

cost_unreg = sum((X*theta - y).^2) * (1 / (2*m));

% We do not regularize theta_0
reg_cost = (lambda / (2*m))* sum(theta(2:end).^2);

J = cost_unreg + reg_cost;

% To calculate the gradient, we take the derivative of the cost function,
% with respect to each theta.
% h(x) - y is the same for each theta

gradient_cost = X*theta - y;

% 1/m is multiplied to each theta
grad = (1/m) .* ( gradient_cost' * X ); % X values are flipped

% Don't calculate regularized term for theta_0
gradient_reg_term = (lambda/m) .* theta(2:end);
grad(2:end) = grad(2:end) + gradient_reg_term'; % Transpose because it needs to be a column vector


% =========================================================================

grad = grad(:);

end
