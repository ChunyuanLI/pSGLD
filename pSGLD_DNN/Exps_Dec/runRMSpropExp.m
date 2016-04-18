%clearvars;
addpath(genpath('.'));
close all
global gpu;
gpu = true;

% if gpu
%     if isempty(gcp('nocreate'))
%             nGPUs = gpuDeviceCount();
%             parpool('local', nGPUs);
%     end
% else
%     if isempty(gcp('nocreate'))
%             parpool('local');
%     end
% end

%savePath = 'exps/';

opts.precision = @single;
opts.flatten = false;
opts.gpu = gpu;
opts.whiten = true;


Ntrain  = 50000; %Subsample
Ntest   = 10000;

%dataFactory = @() Cifar10Data(opts);
data = Cifar10Data(Ntrain,Ntest,opts);


load 'exps/RMSprop40.mat';

exps{1}.data = data;

%mnistExperiments;
%cifarExperiments;
%mnistConvolutionExperiments;
%cifarConvolutionExperiments;


maxEpochs=80;

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
plotAvgWindowSize = 0;
avgWindow = ones(1,plotAvgWindowSize)/plotAvgWindowSize;
for ii=1:numel(exps)
    legendNames = [legendNames;exps{ii}.name];
end
for ii=1:numel(exps)
    exps{ii}.dataChunkSize = 100;
    exps{ii}.randSeed = 102;
    exps{ii}.descentOpts.gpu = gpu;
    exps{ii}.descentOpts.batchSize = 100;
    exps{ii}.descentOpts.epochs = maxEpochs;
    exps{ii}.descentOpts.initialEpochs = 40;
    %exps{ii}.savePath = savePath;
    exps{ii}.reportBatchInterval = 0;
    exps{ii}.run();
    exps{ii}.model.sanitize();
    %exps{ii}.save();
    disp('-----------------');
    set(0,'currentfigure',h);
    if plotAvgWindowSize > 0
        y = filter(avgWindow,1,exps{ii}.results.trainErrors(:));
    else
        y = exps{ii}.results.trainErrors(:);
    end
    line(1:numel(y),y,'color',colors(ii,:));
    legend(legendNames{1:ii});
    drawnow;
end
%data.clear();
% fprintf('Saving...');
% save(resultsPath, 'exps','-v7.3');
% fprintf(' done!\n');
%%

%%
%axis([0 numel(exps{1}.results.trainErrors) 0 200])
