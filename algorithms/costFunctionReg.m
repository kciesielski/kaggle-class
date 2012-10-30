function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
lambdaByM =  lambda / m;
lambdaBy2M =  lambdaByM / 2;
hypoX = sigmoid(X*theta);
J = (sum(-y .* log(hypoX) - ((1 - y) .* log(1 - hypoX))) / m) + (lambdaBy2M * sum(theta(2:size(theta,1)) .^ 2));
grad = (((hypoX - y)' * X) / m) + (theta' * lambdaByM);
grad(1) = ((hypoX - y)' * X(:,1)) / m;
end
