function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

prediction_error = 0;
best_c = 0;
best_s = 0;
for c_index = 1:size(C_vals,1)
    
    for sig_index = 1:size(sigma_vals,1)
        
       current_C = C_vals(c_index);
       current_sigma = sigma_vals(sig_index);

       model = svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma)); 
       
       predictions = svmPredict(model, Xval);
       
       current_error = mean(double(predictions ~= yval));
       
       if (prediction_error == 0)
           prediction_error = current_error;
       elseif (current_error < prediction_error)
           best_c = c_index;
           best_s = sig_index;
           prediction_error = current_error;
       end
        
    end

end

C = C_vals(best_c);
sigma = sigma_vals(best_s);

% =========================================================================

end
