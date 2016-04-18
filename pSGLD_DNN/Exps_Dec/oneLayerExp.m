
clearvars;
addpath(genpath('.'));
close all
gpu = false;

savePath = 'exps/';

N=1000;
% dataFactory = @() ToyData(N);
% N = 60000; %Subsample
dataFactory=@() MnistData(N);
data = dataFactory();


%%
exps = {};
sizes = [data.inSize 200 data.outSize];
%% quick init for plot clarity
NNFactory = @() SimpleNN(sizes, 'SReLU', gpu);
initialization=false;
if initialization
init=GradientDescent('Mnist-3L-GD',NNFactory,dataFactory);
init.descentOpts.gradScale = 0.1;
init.descentOpts.batchSize = 20;
init.descentOpts.epochs=5;
init.run();
initParameters=init.model.getParameters;
% initParameters(:)=0;
NNFactory = @()SimpleNN(sizes, 'SReLU', gpu,initParameters);
end
%%

init.descentOpts.batchSize = N;
ex = GradientDescent('Mnist-3L-GD',NNFactory,dataFactory);
ex.descentOpts.gradScale = .1;
exps{numel(exps)+1} = ex;


ex = SpectralDescentDavid('Mnist-3L-SD-D',NNFactory,dataFactory);
ex.descentOpts.gradScale = 5;
exps{numel(exps)+1} = ex;

maxEpochs=100;

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
for ii=1:numel(exps)
    exps{ii}.gpu = gpu;
    exps{ii}.descentOpts.batchSize = N;
    exps{ii}.descentOpts.epochs = maxEpochs;
    exps{ii}.savePath = savePath;
    exps{ii}.run();
    %ex{ii}.save();
    disp('-----------------');
    line(1:maxEpochs,exps{ii}.results.trainErrors,'color',colors(ii,:));
    legendNames = [legendNames;exps{ii}.name];
    legend(legendNames);
    drawnow;
end
%%