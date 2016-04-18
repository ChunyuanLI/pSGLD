
clearvars;
addpath(genpath('.'));
close all
gpu = false;

savePath = 'exps/';

N = 1000; %Subsample
dataFactory = @() Cifar10Data(N);
data = dataFactory();
visualize = @(x) visualizeCifar(x,true);


%%
exps = {};
use1layerEx=false;
use2layerEx=true;
if use1layerEx
    sizes = [data.inSize data.outSize];
%% quick init for plot clarity
NNFactory = @() SimpleNN(sizes, Sigmoid, gpu);
init=GradientDescent('Mnist-1L-GD',NNFactory,dataFactory);
init.descentOpts.gradScale = 0.2;
init.descentOpts.batchSize = N;
init.descentOpts.epochs=15;
init.run();
initParameters=init.model.getParameters;
% initParameters(:)=0;
NNFactory = @()SimpleNN(sizes, Sigmoid, gpu,initParameters);
%%
ex = GradientDescent('Mnist-1L-GD',NNFactory,dataFactory);
ex.descentOpts.gradScale = 4;
exps{numel(exps)+1} = ex;

ex = SpectralDescentEdo('Mnist-1L-SD-E',NNFactory,dataFactory);
ex.descentOpts.gradScale = 28;
exps{numel(exps)+1} = ex;

ex = SpectralDescentDavid('Mnist-1L-SD-D',NNFactory,dataFactory);
ex.descentOpts.gradScale = 15;
exps{numel(exps)+1} = ex;
end

if use2layerEx
sizes = [data.inSize 250 data.outSize];
NNFactory = @() SimpleNN(sizes, 'Sigmoid', gpu);

ex = VisualizeExperiment('Mnist-2L-GD',NNFactory,dataFactory,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .2;
exps{numel(exps)+1} = ex;

ex = VisualizeExperiment('Mnist-2L-Adagrad',NNFactory,dataFactory,@Adagrad);
ex.descentOpts.learningRate = .02;
exps{numel(exps)+1} = ex;

ex = VisualizeExperiment('Mnist-2L-RMSprop',NNFactory,dataFactory,@RMSprop);
ex.descentOpts.learningRate = .005;
ex.descentOpts.learningRateDecay = .02;
ex.descentOpts.RMSpropDecay = .8;
%exps{numel(exps)+1} = ex;

 ex = VisualizeExperiment('Mnist-2L-SD-E',NNFactory,dataFactory,@SpectralDescentEdo);
 ex.descentOpts.learningRate = 30;
 %exps{numel(exps)+1} = ex;

ex = VisualizeExperiment('Mnist-2L-SD-D',NNFactory,dataFactory,@SpectralDescentDavid);
ex.descentOpts.learningRate = 200;
exps{numel(exps)+1} = ex;

ex = VisualizeExperiment('Mnist-2L-SD-Min',NNFactory,dataFactory,@SpectralDescentMin);
ex.descentOpts.learningRate = 1;
%exps{numel(exps)+1} = ex;

ex = VisualizeExperiment('Mnist-2L-SD-Max',NNFactory,dataFactory,@SpectralDescentMax);
ex.descentOpts.learningRate = 1;
%exps{numel(exps)+1} = ex;

end


maxEpochs=100;

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
for ii=1:numel(exps)
    exps{ii}.visualize = visualize;
    exps{ii}.randSeed = 102;
    exps{ii}.descentOpts.gpu = gpu;
    exps{ii}.descentOpts.batchSize = N;
    exps{ii}.descentOpts.epochs = maxEpochs;
    exps{ii}.savePath = savePath;
    exps{ii}.run();
    %exps{ii}.save();
    disp('-----------------');
    set(0, 'currentfigure', h);
    line(1:maxEpochs,exps{ii}.results.trainErrors,'color',colors(ii,:));
    legendNames = [legendNames;exps{ii}.name];
    legend(legendNames);
    drawnow;
end
%%
return
%%
axis([0 maxEpochs 0 200])
