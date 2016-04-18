function [exps,data,resultsPath] = cifarConvolutionExperiments(expSet)
global gpu;
opts.precision = @single;
opts.flatten = false;
opts.gpu = gpu;
opts.whiten = true;


Ntrain  = 50000; %Subsample
Ntest   = 10000;
data = Cifar10Data(Ntrain,Ntest,opts);

exps = {};

% convolutionSizes = ...
% [5       5  ;       % filter height
%  5       5  ;       % filter width
%  16      16 ;       % #out channels 
%  2       2  ;       % max pooling height
%  2       2  ]';     % max pooling width
% %1st    2nd           layers

if strcmp(expSet,'1L')
    convolutionSizes = [5 5 16 2 2];
    resultsPath = 'exps/cifar-1L-CNN.mat';

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
    ex.descentOpts.learningRate = .01;
    ex.descentOpts.learningRateDecay = .005;
    ex.descentOpts.RMSpropDecay = .95;
    exps{end+1} = ex;

    ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = false;
    exps{end+1} = ex;

    
    ex = SimpleExperiment('RMSSpectral',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = false;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;
    

    ex = SimpleExperiment('AdaSpectral',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = false;
    exps{end+1} = ex;
    
end



if strcmp(expSet,'2L')
    convolutionSizes = [5 5 64 2 2; ...
                         5 5 64 2 2];
    resultsPath = 'exps/cifar-2L-CNN-2nd.mat';
    
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
    ex.descentOpts.learningRate = .01;
    ex.descentOpts.learningRateDecay = .005;
    ex.descentOpts.RMSpropDecay = .95;
    exps{end+1} = ex;

    ex = SimpleExperiment('SSD',NNFactory,data,@SpectralDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = false;
    exps{end+1} = ex;

    ex = SimpleExperiment('RMSSpectral',NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.projectKernels = false;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;
    

    ex = SimpleExperiment('AdaSpectral',NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 1e-3;
    ex.descentOpts.epsilon = 1e-3;
    ex.descentOpts.projectKernels = false;
    exps{end+1} = ex;
end

end
