
clearvars; clc; close all
addpath(genpath('.'));

global gpu;
% try
%     gpuDevice;
%     gpu=true;
% catch
%     gpu = false;
% end
gpu=false;

opts.precision = @single;
opts.flatten = true;
opts.gpu = gpu;

Ntrain  = 60000; %Subsample
Ntest   = 10000;

data      = MnistData(Ntrain,Ntest,opts);
actFunc   = 'ReLU'; % 'Sigmoid',  'Tanh'; 'SReLU';  'ReLU';
inSize    = [28 28 1]; 
convSizes = []; % 0 layers
linSizes  = [400 400 data.outSize]
NNFactory = @() SimpleCNN(inSize,convSizes, linSizes, actFunc);

batchSize = 100; stepsize = 5e-4; %5e-1; % 1e-4 for SGNHTS;   1e-1 for SGLD ; 2e-3; for SGLDADAM
maxEpochs = 100; % Experiment Parameters
C = 50;

%% Setup experiment
exps={};
% 
% ex = SimpleExperiment('SGLD',NNFactory,data,@SGLD);
% ex.descentOpts.learningRate = stepsize;
% ex.descentOpts.learningRateDecay =  0.02;
% ex.descentOpts.weightDecay= 1/Ntrain;
% ex.descentOpts.RMSpropDecay = .99;
% ex.descentOpts.epsilon=1e-5;
% ex.descentOpts.N=Ntrain;
% ex.descentOpts.batchSize = batchSize;
% ex.saveInterval = -1;
% ex.descentOpts.learningRateBlockDecay=0.5;
% ex.descentOpts.learningRateBlock = maxEpochs*1*Ntrain/batchSize;
% ex.descentOpts.burnin= 1*Ntrain/batchSize;
% exps{end+1}=ex;
% 
% ex = SimpleExperiment('SGD',NNFactory,data,@SGD);
% ex.descentOpts.learningRate = stepsize;
% ex.descentOpts.weightDecay= 1/Ntrain;
% ex.descentOpts.learningRateDecay =  0.00;
% ex.descentOpts.RMSpropDecay = .99;
% ex.descentOpts.epsilon=1e-5;
% ex.descentOpts.N=Ntrain;
% ex.descentOpts.batchSize = batchSize;
% ex.descentOpts.learningRateBlockDecay=0.5;
% ex.descentOpts.learningRateBlock = maxEpochs*0.2*Ntrain/batchSize;
% ex.descentOpts.burnin= 1*Ntrain/batchSize;
% exps{end+1}=ex;

% 
ex = SimpleExperiment('SGLD\_RMSprop',NNFactory,data,@SGLD_RMSprop);
ex.descentOpts.learningRate = stepsize;
ex.descentOpts.learningRateDecay =  0.00;
ex.descentOpts.weightDecay= 1/Ntrain;
ex.descentOpts.RMSpropDecay = .99;
ex.descentOpts.epsilon=1e-5;
ex.descentOpts.N=Ntrain;
ex.descentOpts.batchSize = batchSize;
ex.saveInterval = -1;
ex.descentOpts.learningRateBlockDecay=0.5;
ex.descentOpts.learningRateBlock = maxEpochs*0.2*Ntrain/batchSize;
ex.descentOpts.burnin= 0.5*Ntrain/batchSize;
exps{end+1}=ex;

%  
% ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
% ex.descentOpts.learningRate = stepsize;
% ex.descentOpts.weightDecay= 1/Ntrain;
% ex.descentOpts.learningRateDecay =  0.00;
% ex.descentOpts.RMSpropDecay = .99;
% ex.descentOpts.epsilon=1e-5;
% ex.descentOpts.N=Ntrain;
% ex.descentOpts.batchSize = batchSize;
% ex.descentOpts.learningRateBlockDecay=0.5;
% ex.descentOpts.learningRateBlock = maxEpochs*0.2*Ntrain/batchSize;
% ex.descentOpts.burnin= 0.5*Ntrain/batchSize;
% exps{end+1}=ex;






exps=runExperiments(exps,maxEpochs,1);
