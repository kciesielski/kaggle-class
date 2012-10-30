function out = addPolynomialFeatures(X1, X2, X3, degree)


out = ones(size(X1(:,1)));
for i = 2:degree
    for j = 1:i
	for k= 0:j

        out(:, end+1) = (X1.^(i-j-k)).*(X2.^j-k).*(X3.^k);
    end
end

end
