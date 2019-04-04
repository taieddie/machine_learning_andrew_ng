function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% Notes:
% theta = (X_tX)^-1*X_ty , where X_t = transpose of X and (^-1) is the
% inverse.
% ex1_multi adds a column of ones already

X_t = X.';

theta = (inv(X_t*X))*X_t*y;

% -------------------------------------------------------------


% ============================================================

end
