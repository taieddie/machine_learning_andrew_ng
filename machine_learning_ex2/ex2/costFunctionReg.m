function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% We use the original cost and gradient descent function and then add
% the sums we need to get the regularized functions.
[J_regular, grad_regular] = costFunction( theta, X, y );

J_new = J_regular;
% With regularized logistic regression, we need the sum of the thetas for
% each feature squared. We do not include theta 0.
J_new = J_new + ((lambda/(2*m))*sum(theta(2:end).^2));
J = J_new;

grad_new = grad_regular;
% For gradient descent
grad_new(2:end) = grad_new(2:end) + ((lambda/m)*theta(2:end))';
grad = grad_new;



% =============================================================

end
