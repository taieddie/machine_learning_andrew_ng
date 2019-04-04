function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %.

    % theta(j) := theta(j) - learning_rate *  h(theta)(xi) - y(i) * x(i)j
    % h(theta)(xi) = X (matrix) * theta
    
    theta = theta - (alpha .* (1/m) .* ( sum ((X*theta - y) .* X , 1)))';  
    
    %for j = 1:length(theta)
    %    new_theta(j) = theta(j) - alpha * (1/m) * ( sum( (X*theta - y) .* X(:,j)) );
    %end

    %{
    fprintf('Iteration %d, ', iter);
    
    for i = 1:length(theta)
       
        fprintf('i = %d, theta = %.2f ', i, theta(i));
        
    end
    %}
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    if ( (iter > 1) & ( J_history(iter) > J_history(iter-1) ) )
       
        fprintf ('Error: The cost function has increased with alpha = %d, at iteration = %d, new cost = %.2f, old cost = %.2f\n', ...
                 alpha, iter, J_history(iter), J_history(iter-1));
        %Do not exit, return;
    end
    %{
    fprintf (', Cost = %d', J_history(iter));
    
    fprintf ('\n');
    %}
end

end
