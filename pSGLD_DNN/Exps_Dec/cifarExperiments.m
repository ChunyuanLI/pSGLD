function [exps,data,resultsPath] = cifarExperiments(expSet,nonLin,sz)
global gpu;
opts.precision = @single;
opts.flatten = true;
opts.gpu = gpu;
opts.whiten = false;

Ntrain  = 50000; %Subsample
Ntest   = 10000;

data = Cifar10Data(Ntrain,Ntest,opts);
%visualize = @(x) visStuff(x,data.inSize);


resultsPath = sprintf('exps/cifar-%s-%d-FFNN-%s.mat',expSet,sz,nonLin)
exps = {};



if strcmp(expSet,'1L')

sizes = [prod(data.inSize) data.outSize];
NNFactory = @() SimpleNN(sizes, nonLin, gpu);


%%
ex = SimpleExperiment('GD',NNFactory,data,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
exps{end+1} = ex;

ex = SimpleExperiment('Adagrad',NNFactory,data,@Adagrad);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .03;
exps{end+1} = ex;

ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .02;
ex.descentOpts.learningRateDecay = .005;
ex.descentOpts.RMSpropDecay = .95;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-1',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-1;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-2',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-2;
exps{end+1} = ex;



ex = SimpleExperiment('RMSSpectral-6',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-3;
exps{end+1} = ex;



ex = SimpleExperiment('AdaSpectral-1',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-1;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-2',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-2;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-3',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-3;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-4',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-1;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-2',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-2;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-1',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-3;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;


ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
ex.descentOpts.gpu = gpu;
%ex.descentOpts.weightDecay=0.005;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
% exps{end+1} = ex;


ex = SimpleExperiment('SSD-K',NNFactory,data,@SpectralDescent);
ex.descentOpts.gpu = gpu;
%ex.descentOpts.weightDecay=0.005;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
%exps{end+1} = ex;

end

if strcmp(expSet,'2L')

sizes = [prod(data.inSize) sz data.outSize];
NNFactory = @() SimpleNN(sizes, nonLin, gpu);


%%
ex = SimpleExperiment('GD',NNFactory,data,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
%exps{end+1} = ex;

ex = SimpleExperiment('Adagrad',NNFactory,data,@Adagrad);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .03;
%exps{end+1} = ex;

ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .02;
ex.descentOpts.learningRateDecay = .005;
ex.descentOpts.RMSpropDecay = .95;
%exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-1',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-1;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-2',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-2;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-3',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-3;
exps{end+1} = ex;


ex = SimpleExperiment('RMSSpectral-4',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-1;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-5',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-2;
exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral-6',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-3;
exps{end+1} = ex;



ex = SimpleExperiment('AdaSpectral-1',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-1;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-2',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-2;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-3',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-3;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-4',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-1;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-2',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-2;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral-1',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-3;
ex.descentOpts.projectKernels = true;
exps{end+1} = ex;


ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
ex.descentOpts.gpu = gpu;
%ex.descentOpts.weightDecay=0.005;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
% exps{end+1} = ex;


ex = SimpleExperiment('SSD-K',NNFactory,data,@SpectralDescent);
ex.descentOpts.gpu = gpu;
%ex.descentOpts.weightDecay=0.005;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.projectKernels = true;
%exps{end+1} = ex;

end

if strcmp(expSet,'3L')

sizes = [prod(data.inSize) sz data.outSize];
NNFactory = @() SimpleNN(sizes, nonLin, gpu);

ex = SimpleExperiment('GD',NNFactory,data,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
%exps{end+1} = ex;

ex = SimpleExperiment('Adagrad',NNFactory,data,@Adagrad);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .03;
%exps{end+1} = ex;

ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .02;
ex.descentOpts.learningRateDecay = .005;
ex.descentOpts.RMSpropDecay = .95;
%exps{end+1} = ex;

ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 0.1;
ex.descentOpts.learningRateDecay = .2;
%exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral',NNFactory,data,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.epsilon = 1e-3;
%exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectral',NNFactory,data,@RMSSpectral3);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 3e-3;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.epsilon = 1e-3;
%exps{end+1} = ex;

ex = SimpleExperiment('RMSSpectralNoBias',NNFactory,data,@RMSSpectral3NoBias);
ex.descentOpts.gpu = gpu;
ex.descentOpts.RMSpropDecay = .95;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.learningRateDecay = .2;
%ex.descentOpts.projectKernels = true;
ex.descentOpts.epsilon = 1e-3;
exps{end+1} = ex;


ex = SimpleExperiment('AdaSpectralNoBias',NNFactory,data,@AdaSpectralNoBias);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
ex.descentOpts.epsilon = 1e-3;
%ex.descentOpts.projectKernels = true;
exps{end+1} = ex;

end


