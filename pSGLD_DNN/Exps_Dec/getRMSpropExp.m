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

[exps,data,resultsPath] = cifarConvolutionExperiments('2L');

resultsPath = 'exps/RMSprop40.mat';
exps = exps(3);

%mnistExperiments;
%cifarExperiments;
%mnistConvolutionExperiments;
%cifarConvolutionExperiments;


maxEpochs=40;

h = figure;
colors = distinguishable_colors(numel(exps));
legendNames = {};
plotAvgWindowSize = 0;
avgWindow = ones(1,plotAvgWindowSize)/plotAvgWindowSize;
for ii=1:numel(exps)
    legendNames = [legendNames;exps{ii}.name];
end
for ii=1:numel(exps)
    exps{ii}.dataChunkSize = 6000;
    exps{ii}.randSeed = 102;
    exps{ii}.descentOpts.gpu = gpu;
    exps{ii}.descentOpts.batchSize = 2000;
    exps{ii}.descentOpts.epochs = maxEpochs;
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
data.clear();
fprintf('Saving...');
save(resultsPath, 'exps','-v7.3');
fprintf(' done!\n');
%%

%%
%axis([0 numel(exps{1}.results.trainErrors) 0 200])
