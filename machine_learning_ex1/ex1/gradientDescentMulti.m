function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % To explain the equation below - 
    % X*theta will give u h(x(i)). h(x(i)) is really theta0x0 + theta1x1 +
    % theta2x2, etc. Note: x0 is just 1. So if we multiple X*theta, you
    % really get theta0x0 + theta1x1,etc. That is the correct sum. 
    % That "hypothesis" value is subtracted by the actual value, which we can call the cost.
    % This column vector is really h(x(i)) - y(i). Then, 
    % we multiply it by X because each feature given
    % i and j, multiply the cost. The last step is basically when we sum
    % up each training set and then divide by m to get the average cost. 
    % That is the reason why there is a sum in the formula. The sum will
    % take the sum column wise. 
    
    % Alternate way to see it,
    % We loop through each feature, for each feature we calculate the cost
    % using the training samples and then divide by the number training
    % samples. Looks very similar to the cost, except we multiply each
    % training cost by the training value.
    
    theta = theta - (alpha .* (1/m) .* ( sum ((X*theta - y) .* X, 1 )))';

    % ============================================================

    fprintf('Iteration %d, ', iter);
    
    for i = 1:length(theta)
       
        fprintf('i = %d, theta = %.2f ', i, theta(i));
        
    end
    
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    
    if ( (iter > 1) & ( J_history(iter) > J_history(iter-1) ) )
       
        fprintf ('Error: The cost function has increased with alpha = %d, at iteration = %d, new cost = %.2f, old cost = %.2f\n\n', ...
                 alpha, iter, J_history(iter), J_history(iter-1));
        % Do not exit, return;
    end
    
    fprintf (', Cost = %d', J_history(iter));
    
    fprintf ('\n');
    
end

end
