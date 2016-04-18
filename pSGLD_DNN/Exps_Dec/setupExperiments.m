function exps=setupExperiments(NNFactory,dataFactory,pen)
exps = {};
%%
ex = SimpleExperiment('RMSSpec',NNFactory,dataFactory,@RMSSpectral5);
convolutionSizes = [5 5 32 2 2];
ex.descentOpts.learningRate = .001;
ex.descentOpts.learningRateDecay = .2;
ex.descentOpts.RMSpropDecay = .99;
ex.descentOpts.epsilon=1e-3;
exps{numel(exps)+1} = ex;

%%
ex = SimpleExperiment('RMSprop',NNFactory,dataFactory,@RMSprop);
ex.descentOpts.learningRate = .01;
ex.descentOpts.learningRateDecay = 0;
ex.descentOpts.RMSpropDecay = .99;
ex.descentOpts.epsilon=1e-3;
exps{numel(exps)+1} = ex;
if false
    %%
ex = SimpleExperiment('AdaSpec',NNFactory,dataFactory,@AdaSpectral);
ex.descentOpts.learningRate = .01;
ex.descentOpts.epsilon=1e-3;
exps{numel(exps)+1} = ex;
    %%
    ex = SimpleExperiment('SGD',NNFactory,dataFactory,@GradientDescent);
    ex.descentOpts.learningRate = .04;
    ex.descentOpts.learningRateDecay = .5;
    exps{numel(exps)+1} = ex;
    
    %%
    ex = SimpleExperiment('Adagrad',NNFactory,dataFactory,@Adagrad);
    ex.descentOpts.learningRate = .01;
    exps{numel(exps)+1} = ex;
    % %%
    % ex = SimpleExperiment('SSD',NNFactory,dataFactory,@approxRandSpectralDescentDavid);
    % data = dataFactory();
    % ex.descentOpts.learningRateDecay = .5;
    % ex.descentOpts.learningRateOffset=10;
    % ex.descentOpts.learningRate = 2;
    % ex.descentOpts.data = data;
    % exps{numel(exps)+1} = ex;
end
%%  Add penalty
if nargin>=3
    for ex=1:numel(exps);
        exps{ex}.descentOpts.weightDecay=pen;
    end
end