

clearvars;
addpath(genpath('.'));
close all
gpu = false;

savePath = 'exps/';


% mnistExperiments;
%cifarExperiments;
exps = {};

N = 5000; %Subsample
dataFactory = @() MnistData(N);
data = dataFactory();
visualize = @(x) visualizeMnist(x,true);
sizes = [data.inSize 100 data.outSize];
NNFactory = @() SimpleNN(sizes, 'SReLU', gpu);

ex = SaddleExperiment('Mnist-2L-GD',NNFactory,dataFactory,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .3;
ex.descentOpts.learningRateDecay = 0;
exps{numel(exps)+1} = ex;
maxEpochs=40;
colors = distinguishable_colors(numel(exps));
legendNames = {};
ex.randSeed = 102;
ex.descentOpts.gpu = gpu;
ex.descentOpts.batchSize = N;
ex.descentOpts.epochs = maxEpochs;
ex.savePath = savePath;
ex.run();
%%
% return
%%
clc
startValue=ex.getValue;
P=ex.model.getParameters;
layer=1;
G=ex.getGrad(layer);
% steps=linspace(0,.5);
% vals=ex.evalLine(G,-steps);
GR=reshape(G,sizes(1)+1,sizes(2));
[U,S,V]=svd(GR,0);S=diag(S);
D1=U(:,1)*V(:,1)';D1=D1(:);
D2=U(:,2)*V(:,2)';D2=D2(:);
N1=8;N2=8;
steps1=linspace(-.5,1,N1)*S(1);
steps2=linspace(-.5,1,N2)*S(2);
gridVals=ex.evalGrid(D1,D2,-steps1,-steps2);
contour(steps1',steps2,gridVals,15),colormap bone
xlabel('\lambda_1')
ylabel('\lambda_2')
hold on
tGD=1/S(1);
plot([0 tGD*S(1)], [0  tGD*S(2)],'k-')
tSD=1/S(1);
plot([0 tGD*S(1)], [0  tGD*S(1)],'m-')
hold off
% imagesc(steps1',steps2,gridVals),colorbar









