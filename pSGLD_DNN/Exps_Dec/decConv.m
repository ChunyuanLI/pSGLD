addpath(genpath('.'))
N=60000;
close all;

try
    gpuDevice;
    gpu=1;
catch
    gpu=false;
end

opts.gpu = gpu;
opts.precision = @single;
opts.flatten = false;

% dataFactory = @() Cifar10Data(N,opts);
dataFactory = @() MnistData(N,opts);
data = dataFactory();
visualize = @(x) visualizeCifar(x,true);

exps = {};


% convolutionSizes = ...
%     [5       5  ;       % filter width
%     5       5  ;       % filter height
%     20      50 ;       % #out channels
%     2       2  ;       % max pooling width
%     2       2  ]';     % max pooling height
%1st    2nd           layers

convolutionSizes = [5 5 32 2 2];
% convolutionSizes = [];

linearSizes = [200 data.outSize]; % first linear inSize will be set to the outputsize of the last convolutional layer.
% sizes = [data.inSize 100 data.outSize];
NNFactory = @() SimpleCNN(data.inSize,convolutionSizes,linearSizes,'SReLU', gpu);

ex = ConvolutionExperiment('Rmsprop',NNFactory,dataFactory,@RMSprop);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .005;
%exps{end+1} = ex;

ex = ConvolutionExperiment('RMSSpectral',NNFactory,dataFactory,@RMSSpectral2);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-2;
%exps{end+1} = ex;

ex = SimpleExperiment('AdaSpectral',NNFactory,dataFactory,@AdaSpectral);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = 1e-3;
exps{end+1} = ex;




ex = ConvolutionExperiment('GD',NNFactory,dataFactory,@GradientDescent);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .001;
%exps{end+1} = ex;



ex = SimpleExperiment('AdaGrad',NNFactory,dataFactory,@Adagrad);
ex.descentOpts.gpu = gpu;
ex.descentOpts.learningRate = .003;
exps{end+1} = ex;



pen=1e-4;
    for ex=1:numel(exps);
    exps{ex}.descentOpts.weightDecay=pen;
    end


maxEpochs=1;
colors = distinguishable_colors(numel(exps));
legendNames = {};
times=[];
for ii=1:numel(exps);
    exps{ii}.descentOpts.batchSize = 1000;
    exps{ii}.descentOpts.epochs = maxEpochs;
    tic
    exps{ii}.run();
    times(ii)=toc,
    exps{ii}.data=[];
    exps{ii}.model.sanitize();
    %     exps{ii}.save();
    disp('-----------------');
    plotting=false;
    if plotting
%         set(0,'currentfigure',h);
        plotall=false;
        if plotall
            y=exps{ii}.results.trainErrors;nP=size(y,1);y=y';y=y(:);
            line(linspace(1/nP,maxEpochs,numel(y)),y,'color',colors(ii,:));
        else
            y=exps{ii}.results.trainErrors;nP=size(y,1);y=y';
            figure(1)
            hold on
            renorm=maxEpochs./exps{1}.results.times(end);
            plot(exps{ii}.results.times.*renorm,mean(y),'.-','color',colors(ii,:));
            axis auto;
            Q=axis;
            Q(2)=maxEpochs;
            axis(Q);
            figure(2)
            hold on
            plot(exps{ii}.results.times.*renorm,exps{ii}.results.testAccuracy,'.-','color',colors(ii,:));
            axis auto;
            Q=axis;
            Q(2)=maxEpochs;
            axis(Q);
        end
        
        figure(1);
        legendNames = [legendNames;exps{ii}.name];
        legend(legendNames);
        set(gca,'yscale','log')
        
        figure(2)
        %         legendNames = [legendNames;exps{ii}.name];
        legend(legendNames);
        drawnow;
    end
    %save Results/decConv0 exps
    save('exps/decConv.mat', 'exps','-v7.3');
end
return
%%
figure(2)
set(gca,'FontSize',20)
xlabel('Normalized Time')
ylabel('Validation Set Performance')
title('MNIST, 1 Conv Layer, 1 Linear Layer')
legend(legendNames,'location','southeast');
orient landscape
%print -dpdf ~/Dropbox/simpleCNN_validation
print -dpdf exps/decConv.pdf
