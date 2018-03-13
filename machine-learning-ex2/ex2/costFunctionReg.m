function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


y0 = log(sigmoid(theta' * X'))'.* -y;
y1 = log(1 - sigmoid(theta' * X'))'.* (1-y);
reg = (theta(2:n)' * theta(2:n)) * lambda / 2 / m;

J = sum(y0-y1)/m + reg;

% grad
grad = ((sigmoid(theta' * X') - y') * X)'/m;
grad(2:n) = grad(2:n) + (theta(2:n) * lambda / m);
grad = round(grad.*10000) ./ 10000;

end

% =============================================================


