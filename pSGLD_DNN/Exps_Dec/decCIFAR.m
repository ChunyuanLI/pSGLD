function [exps,data,resultsPath] = decCIFAR(expSet)
global gpu;
opts.precision = @single;
opts.flatten = false;
opts.gpu = gpu;
opts.whiten = true;


Ntrain  = 50000; %Subsample
Ntest   = 10000;
data = Cifar10Data(Ntrain,Ntest,opts);

exps = {};


convolutionSizes = [5 5 32 2 2; ...
    5 5 32 2 2];
resultsPath = 'exps/decCIFAR.mat';
% convolutionSizes=[5 5 32 2 2];
linearSizes = [100 100 data.outSize];
NNFactory = @() SimpleCNN(data.inSize,convolutionSizes,linearSizes,'ReLU', gpu);


ex = SimpleExperiment('RMSSpectral',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 2e-4;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.weightDecay=1e-3;
ex.descentOpts.projectKernels = false;
ex.descentOpts.epsilon = 1e-4;
exps{end+1} = ex;

