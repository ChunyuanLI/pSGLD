
clearvars;
addpath(genpath('.'));
close all
gpu = false;

savePath = 'exps/';

N = 10000; %Subsample
dataFactory = @() MnistData(N);
data = dataFactory();

sizes = [data.inSize 200 100 50 data.outSize];
NNFactory = @() SimpleNN(sizes, 'Sigmoid', gpu);

exps = {};

ex = StepSizeTrackingExperiment('Mnist-2L-SD-M',NNFactory,dataFactory,@SpectralDescentMisc);
ex.descentOpts.learningRate = 2;
ex.descentOpts.data = data;
%exps{numel(exps)+1} = ex;

ex = StepSizeTrackingExperiment('Mnist-2L-SD-D',NNFactory,dataFactory,@SpectralDescentDavid);
ex.descentOpts.learningRate = 2;
ex.descentOpts.data = data;
exps{numel(exps)+1} = ex;

ex = StepSizeTrackingExperiment('Mnist-2L-SD-approx',NNFactory,dataFactory,@approxSpectralDescentDavid);
ex.descentOpts.learningRate = 2;
ex.descentOpts.data = data;
exps{numel(exps)+1} = ex;

ex = StepSizeTrackingExperiment('Mnist-2L-SD-randapprox',NNFactory,dataFactory,@approxRandSpectralDescentDavid);
ex.descentOpts.learningRate = 2;
ex.descentOpts.data = data;
exps{numel(exps)+1} = ex;


maxEpochs=5;

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
plotAvgWindowSize = 10;
avgWindow = ones(1,plotAvgWindowSize)/plotAvgWindowSize;
for ii=1:numel(exps)
%    exps{ii}.visualize = visualize;
    exps{ii}.randSeed = 102;
    exps{ii}.descentOpts.gpu = gpu;
    exps{ii}.descentOpts.batchSize = 100;
    exps{ii}.descentOpts.epochs = maxEpochs;
    exps{ii}.savePath = savePath;
    exps{ii}.reportBatchInterval = 60;
    exps{ii}.run();
    %exps{ii}.save();
    disp('-----------------');
    set(0,'currentfigure',h);
    if plotAvgWindowSize > 0
        y = filter(avgWindow,1,exps{ii}.results.trainErrors(:));
    else
        y = exps{ii}.results.trainErrors(:);
    end
    line(1:numel(y),y,'color',colors(ii,:));
    legendNames = [legendNames;exps{ii}.name];
    legend(legendNames);
    drawnow;
end
%%
return
%%
axis([0 numel(exps{1}.results.trainErrors) 0 200])
