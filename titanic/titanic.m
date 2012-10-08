%% Initialization
clear ; close all; clc

file = csvread('./data/train.csv');
fprintf("CSV read succesfully\r\n");

%% Dirty Prototype with Logistic Regression
X = file(2:end, [2, 13, 5, 6, 7, 9]);
y = file(2:end, 1);
X = featureNormalize(X);

additionalXCols = mapFeature(X(:,1), X(:, 2),X(:, 3), 6);
X = [X additionalXCols]; %% additional polynomial features to increase variance

%% Splitting 60-20-20 % to get cross-validation set and test set
fullSetSize = size(X,1);
m = round(fullSetSize * 0.6);
mval = round((fullSetSize - m) / 2.0);
mtest = fullSetSize - m - mval;

Xval = X(m + 1 : m + mval, :);
yVal = y(m + 1 : m + mval, :);

Xtest = X(m + mval +1: fullSetSize, :);
yTest = y(m + mval +1: fullSetSize, :);

X = X(1:m, :);
y = y(1:m, :);
%% Data is ready to train, let's try lgostic regression
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda
lambda = 0;
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 10000);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Compute accuracy on our training set
p = predict(theta, X, 0);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% Compute accuracy on cross-validation set
pVal = predict(theta, Xval, 0);
fprintf('CVal Accuracy: %f\n', mean(double(pVal == yVal)) * 100);
% Compute accuracy on training set
ptest = predict(theta, Xtest, 0);
fprintf('Ctest Accuracy: %f\n', mean(double(ptest == yTest)) * 100);
% Predict for the Kaggle test set
fileFinalTest = csvread('./data/test.csv');
fprintf("Final test CSV read succesfully\r\n");
%% Dirty Prototype with Logistic Regression
Xf = fileFinalTest(2:end, [2, 13, 5, 6, 7, 9]);

X_normF = featureNormalize(Xf);
additionalXColsF = mapFeature(X_normF(:,1), X_normF(:, 2),X_normF(:, 3), 6);
Xf = [X_normF additionalXColsF]; %% additional polynomial features to increase variance
pFinal = predict(theta, Xf,0.0);
csvwrite('./out/result.csv',pFinal);
