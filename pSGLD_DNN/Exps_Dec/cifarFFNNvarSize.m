function [exps,data,resultsPath] = cifarFFNNvarSize(nonLin)
global gpu;
opts.precision = @single;
opts.flatten = true;
opts.gpu = gpu;
opts.whiten = true;

Ntrain  = 50000; %Subsample
Ntest   = 10000;

data = Cifar10Data(Ntrain,Ntest,opts);
%visualize = @(x) visStuff(x,data.inSize);


resultsPath = 'exps/cifar-FFNN-VarSize-Rest.mat';
exps = {};
layerSizes = 5:5:50;


for sz=layerSizes

    sizes = [prod(data.inSize) sz data.outSize];
    NNFactory = @() SimpleNN(sizes, nonLin, gpu);

    ex = SimpleExperiment(sprintf('SGD-%d',sz),NNFactory,data,@GradientDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    exps{end+1} = ex;

    ex = SimpleExperiment(sprintf('Adagrad-%d',sz),NNFactory,data,@Adagrad);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .03;
    exps{end+1} = ex;

    ex = SimpleExperiment(sprintf('RMSprop-%d',sz),NNFactory,data,@RMSprop);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = .02;
    ex.descentOpts.learningRateDecay = .005;
    ex.descentOpts.RMSpropDecay = .95;
    exps{end+1} = ex;

    ex = SimpleExperiment(sprintf('SSD-%d',sz),NNFactory,data,@SpectralDescent);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 0.1;
    ex.descentOpts.learningRateDecay = .2;
    exps{end+1} = ex;

    ex = SimpleExperiment(sprintf('AdaSpectral-%d',sz),NNFactory,data,@AdaSpectral);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.learningRate = 3e-3;
    ex.descentOpts.epsilon = 1e-3;
    exps{end+1} = ex;

    ex = SimpleExperiment(sprintf('RMSSpectral-%d',sz),NNFactory,data,@RMSSpectral3);
    ex.descentOpts.gpu = gpu;
    ex.descentOpts.RMSpropDecay = .95;
    ex.descentOpts.learningRate = 3e-3;
    ex.descentOpts.learningRateDecay = .2;
    ex.descentOpts.epsilon = 1e-3;
    %exps{end+1} = ex;

end

end


