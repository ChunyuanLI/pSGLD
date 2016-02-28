function [ data ] = sgld_rmsprop( gradU, eta, L, x, V )
%% SGLD using gradU, for L steps, starting at position x, use SGFS way to take noise gradient level into account, 
%% return data: array of positions
m = length(x); rmsprop = 0.9999; % 0.999;
data = zeros( m, L );
beta = V * eta * 0.5;
if  beta > 1
    error('too big eta');
end

for t = 1 : L
    if t == 1
        tgrad2 = gradU(x).^2;
    else
        tgrad2 = rmsprop*tgrad2 + (1-rmsprop)*gradU(x).^2;
    end
    M = (sqrt(tgrad2) + 1e-4); 
    sigma = sqrt( 2 * eta );
    dx = - gradU( x ) ./M  * eta + randn(2,1)./ sqrt(M) * sigma;
    x = x + dx;
    data(:,t) = x;

end

end
