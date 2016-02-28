%% This file produces Figure 1a, running trace of SGLD and pSGLD
% revised based on the code of SGHMC by Tianqi Chen

clear all; clc; close all;
global covS;
global invS;
V = 1;
% covariance matrix
rho = 0.0;
covS = [ 0.16, rho; rho, 1 ];
invS = inv( covS );
% intial x
x = [0;0];

% this is highest value tried so far for SGLD that does not diverge
etaSGLD = 0.3;%0.18;


% number of steps 
L = 20000;
nset = 5;
probUMap = @(X,Y) exp( - 0.5 *( X .* X * invS(1,1) + 2 * X.*Y*invS(1,2) + Y.* Y *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));   
funcU = @(x) 0.5 * x'*invS*x;
gradUTrue = @(x) invS * x;
gradUNoise = @(x) invS * x  + 0.1*randn(2,1);

% set random seed
randn( 'seed',20 );

% do multiple experiment, record each sample

for i = 1 : nset
    eta = etaSGLD * (0.8^(i-1));
    dsgld = sgld( gradUNoise, eta, L, x, V );
    covESGLD(:,:,i) = dsgld * dsgld' / L;
    meanESGLD(:,i) = mean( dsgld, 2 );
    SGLDeta(i) = eta;
    SGLDauc(i) = mean(aucTime( dsgld, 1 ));
end

for i = 1 : nset
    eta = etaSGLD * (0.8^(i-1));
    dsgld = sgld_rmsprop( gradUNoise, eta, L, x, V );
    rmsprop_covESGLD(:,:,i) = dsgld * dsgld' / L;
    rmsprop_meanESGLD(:,i) = mean( dsgld, 2 );
    rmsprop_SGLDeta(i) = eta;
    rmsprop_SGLDauc(i) = mean(aucTime( dsgld, 1 ));
end


save pre_sgld_cmpdata.mat;
drawcmp_sgld;

