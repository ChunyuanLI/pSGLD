function [exps,data,resultsPath] = mnistConvolutionExperiments(expSet)
global gpu;
opts.precision = @single;
opts.flatten = false;
opts.gpu = gpu;
opts.whiten = false;


Ntrain  = 60000; %Subsample
Ntest   = 10000;
data = MnistData(Ntrain,Ntest,opts);

exps = {};

% convolutionSizes = ...
% [5       5  ;       % filter height
%  5       5  ;       % filter width
%  16      16 ;       % #out channels 
%  2       2  ;       % max pooling height
%  2       2  ]';     % max pooling width
% %1st    2nd           layers

if strcmp(expSet,'1L')

    convolutionSizes = [5 5 64 2 2];
    resultsPath = 'exps/mnist-1L-CNN.mat';

    linearSizes = [300 data.outSize];
    NNFactory = @() SimpleCNN(data.inSize,convolutionSizes,linearSizes,'ReLU', gpu);

    %%
    ex = SimpleExperiment('GD',NNFactory,data,@GradientDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    %exps{end+1} = ex;

    ex = SimpleExperiment('Adagrad',NNFactory,data,@Adagrad);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .01;
    %exps{end+1} = ex;

    ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .08;
    ex.descentOpts.learningRateDecay = .005;
    ex.descentOpts.RMSpropDecay = .95;
    %exps{end+1} = ex;


    ex = SimpleExperiment('RMSSpectral-a',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;
    
    ex = SimpleExperiment('RMSSpectral-b',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-4;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;


    ex = SimpleExperiment('AdaSpectral-a',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;
    
    ex = SimpleExperiment('AdaSpectral-b',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-4;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;
    
    
    ex = SimpleExperiment('RMSSpectralNoBias',NNFactory,data,@RMSSpectral3NoBias);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;


    ex = SimpleExperiment('AdaSpectralNoBias',NNFactory,data,@AdaSpectralNoBias);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;

end

if strcmp(expSet,'2L')

    convolutionSizes = [5 5 64 2 2 ; ...
                         5 5 64 2 2];
                     
    resultsPath = 'exps/mnist-2L-CNN.mat';

    linearSizes = [300 data.outSize];
    NNFactory = @() SimpleCNN(data.inSize,convolutionSizes,linearSizes,'ReLU', gpu);

    %%
    ex = SimpleExperiment('GD',NNFactory,data,@GradientDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    exps{end+1} = ex;

    ex = SimpleExperiment('Adagrad',NNFactory,data,@Adagrad);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .01;
    exps{end+1} = ex;

    ex = SimpleExperiment('RMSprop',NNFactory,data,@RMSprop);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .08;
    ex.descentOpts.learningRateDecay = .005;
    ex.descentOpts.RMSpropDecay = .95;
    exps{end+1} = ex;


    ex = SimpleExperiment('RMSSpectral-a',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;
    
    ex = SimpleExperiment('RMSSpectral-b',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-4;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;


    ex = SimpleExperiment('AdaSpectral-a',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;
    
    ex = SimpleExperiment('AdaSpectral-b',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-4;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;
    
    
    ex = SimpleExperiment('RMSSpectralNoBias',NNFactory,data,@RMSSpectral3NoBias);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = true;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;


    ex = SimpleExperiment('AdaSpectralNoBias',NNFactory,data,@AdaSpectralNoBias);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = true;
    exps{end+1} = ex;
end

end