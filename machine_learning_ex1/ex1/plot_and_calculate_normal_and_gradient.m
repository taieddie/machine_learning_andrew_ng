
% Test different learning rates.

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
hold on;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

alpha = 0.1;
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);

alpha = 1;
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);

% Given the size of the house and the number of bedrooms, 
% predict the price.

% We now have the best theta values. Lets find y
X_to_predict = [1650, 3] % House with 1650 square feet and 3 bedrooms

% We have no normalize the features...
[X_to_predict mu sigma] = featureNormalize(X_to_predict);

X_to_predict = [1 X_to_predict];

% Alpha 0.1 looks good, lets use it!
alpha = 0.1;
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

price_gradient_descent = X_to_predict*theta;

fprintf ('With gradient descent, predicted price of a 1650 square feet house with 3 bedrooms = %.2f\n', price_gradient_descent);

theta_normal_equ = normalEqn(X,y);

price_normal_eq = X_to_predict*theta_normal_equ;

fprintf ('With normal equation, predicted price of a 1650 square feet house with 3 bedrooms = %.2f\n', price_normal_eq);






