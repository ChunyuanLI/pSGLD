function [Xwh, mu, invMat, whMat] = whiten(X,epsilon)
%function [X,mu,invMat] = whiten(X,epsilon)
%
% ZCA whitening of a data matrix (make the covariance matrix an identity matrix)
%
% WARNING
% This form of whitening performs poorly if the number of dimensions are
% much greater than the number of instances
%
%
% INPUT
% X: columns are the instances, rows are the features
% epsilon: small number to compensate for nearly 0 eigenvalue [DEFAULT = 1e-4]
%
% OUTPUT
% Xwh: whitened data, columns are instances, rows are features
% mu: mean of each feature of the orginal data
% invMat: the inverse data whitening matrix
% whMat: the whitening matrix


if ~exist('epsilon','var')
    epsilon = 1e-4;
end

mu = mean(X,2); 
X = bsxfun(@minus, X, mu);
A = X*X';
[~,D,V] = svd(A);
whMat = sqrt(size(X,1)-1)*V*sqrtm(inv(D + eye(size(D))*epsilon))*V';
Xwh = whMat*X;  
invMat = pinv(whMat);

end