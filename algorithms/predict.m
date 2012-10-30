function p = predict(theta, X, treshold)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);
for i = 1:m
	p(i)=round(sigmoid(X(i,:)*theta) - treshold);
end
end
