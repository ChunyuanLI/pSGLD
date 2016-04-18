function exps=runExperiments(exps,maxEpochs,plotting)
if nargin<3
    plotting=true;
end

if plotting
    h = figure(1);
    h2 = figure(2);
end
colors = distinguishable_colors(numel(exps));
legendNames = {};
times=[];
savePath = 'Result_MNIST/';
for ii=1:numel(exps)
    
    try
        exps{ii}.visualize = visualize;
    end
    exps{ii}.randSeed = 101;
%    exps{ii}.descentOpts.gpu = gpu;
%     if ii==1
%         exps{ii}.descentOpts.batchSize=  500;
%     else
%         exps{ii}.descentOpts.batchSize = 500;
%     end
    exps{ii}.descentOpts.epochs = maxEpochs;
    exps{ii}.savePath = savePath;
    
    tic
    exps{ii}.run();
    times(ii)=toc;
    
%     exps{ii}.save();
    disp('-----------------');
    if plotting
        set(0,'currentfigure',h);
        plotall=false;
        if plotall
            y=exps{ii}.results.trainErrors;nP=size(y,1);y=y';y=y(:);
            line(linspace(1/nP,maxEpochs,numel(y)),y,'color',colors(ii,:));
        else
            y=exps{ii}.results.trainErrors;nP=size(y,1);y=y;
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
end
