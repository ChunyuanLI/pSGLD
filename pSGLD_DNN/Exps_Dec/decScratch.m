
clearvars;
addpath(genpath('.'));
close all
try
    gpuDevice;
    gpu=true;
catch
    gpu = false;
end

N = 60000; %Subsample
dataFactory = MnistData(N);
data = dataFactory();
visualize = @(x) visualizeMnist(x,true);
% NNFactory = @() SimpleNN(sizes, 'SReLU', gpu);
actFunc='SReLU';
inSize=[28 28 1];
convSizes=[];
linSizes=[200 200 200 200 data.outSize];
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
