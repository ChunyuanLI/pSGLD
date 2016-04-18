
clearvars;
addpath(genpath('.'));
close all
global gpu;
try
    gpuDevice;
    gpu=1;
catch
    gpu=false;
end

opts.gpu = gpu;
opts.precision = @single;
opts.flatten = false;

N = 60000; %Subsample
Ntest = 10000;
dataFactory = MnistData(N,Ntest,opts);
data = dataFactory();
visualize = @(x) visualizeMnist(x,true);
sizes = [data.inSize 200 data.outSize];
% NNFactory = @() SimpleNN(sizes, 'SReLU', gpu);
actFunc='SReLU';
inSize=[28 28 1];
% convSizes=[];
% convSizes = [5 5 32 2 2];
convSizes = ...
    [5       5  ;       % filter width
    5       5  ;       % filter height
    20      50 ;       % #out channels
    2       2  ;       % max pooling width
    2       2  ]';     % max pooling height
% 1st    2nd           layers
linSizes=[200 data.outSize];
NNFactory = @() SimpleCNN(inSize,convSizes, linSizes, actFunc);
%%
pen=1e-6;
exps=setupExperiments(NNFactory,dataFactory,pen);
%%
maxEpochs=100;
%%
exps=runExperiments(exps,maxEpochs);
%%
figure(1)
hold off
xlabel('Normalized Time')
ylabel('Training log-likelihood')
title(['MNIST, Network Size ',sprintf('%d ',sizes(2:end-1))]);
figure(2)
xlabel('Normalized Time')
ylabel('Test Set Accuracy')
title(['MNIST, Network Size ',sprintf('%d ',sizes(2:end-1))]);
return
%%
axis([0 maxEpochs 0 200])
%%
figure(2)
orient landscape
print -dpdf ~/testSetAccuracy
