function plotFeedForwardResults()
    %% MNIST
    files = {'mnist-3L-300-FFNN-Sigmoid-fresh.mat'};
    relabel = { {'GD','SGD'}  {'Adagrad','AdaGrad'}  {'RMSSpectral', 'RMSspectral'}};
    select = {'SGD','AdaGrad','RMSprop','SSD','AdaSpectral','RMSspectral'};
    saveAs = 'exps/MNIST-3Layer-FFNN-Sigmoid-trainErrors.pdf';
    batchesPerEpoch = 30;
    
    opts.xlabel = 'Normalized time';
    opts.ylabel = '$\log p(\mathbf{v})$';
    opts.title = 'MNIST, 3-Layer NN';
    opts.fontSize = 25;
    opts.markerFrequency = 4;
    opts.yscale = 'log';
    
    
    structure = Curves.aggregate(files,@Curves.fileLoader); % load file contents
    structure = Curves.flatten(structure);
    
    accessor = getExpsAccessor('trainErrors');
	curves = Curves.aggregate(structure,accessor);
    
    curves = Curves.findReplace(curves,'label',relabel);
    curves = Curves.select(curves,'label',select);
    curves = Curves.process(curves,@(x) x*batchesPerEpoch,'x');
    curves = Curves.process(curves,@(y) -y,'y');
    
    curves = Curves.process(curves,@(y) gather(y), 'y');
    if size(curves{1}.y,1)>size(curves{1}.y,2)
        curves = Curves.process(curves,@(y) mean(y,2), 'y');
    else
        curves = Curves.process(curves,@(y) mean(y,1)', 'y');
    end
    minLength = min(cell2mat(Curves.aggregate(curves,@(curve) length(curve.y))));
    curves = Curves.process(curves,@(y) y(1:minLength), 'y');
    curves = Curves.process(curves,@(x) x(1:minLength), 'x');
    
    sgd = curves{1};
    curves{4}.avgTime = sgd.avgTime*1.1;
    curves = Curves.process(curves,@(exp) Curves.normalizeX(exp,sgd.avgTime));
    opts.xlim = [0 max(sgd.x)];
    
    x = Curves.aggregate(curves,'x');
    y = Curves.aggregate(curves,'y');
    labels = Curves.aggregate(curves,'label');

	prettyPlotting(x,y,labels,opts);
    print('-dpdf',saveAs); 

    
    
    %% CIFAR
    
    
    files = {'cifar-FFNN-VarSize.mat'};
    saveAs = 'exps/Cifar10-2Layer-FFNN-Sigmoid-trainErrors.pdf';
    opts.title = 'Cifar-10, 2-Layer NN';  
    batchesPerEpoch = 25;
    
    structure = Curves.aggregate(files,@Curves.fileLoader); % load file contents
    structure = Curves.flatten(structure);
    
    accessor = getExpsAccessor('trainErrors');
	curves = Curves.aggregate(structure,accessor);
    
    curves = Curves.filter(curves, @(curve) Curves.endsWith(curve.label,'250L'));
    curves = Curves.process(curves,@(label) label(1:end-5),'label');
    
    curves = Curves.process(curves,@(x) x*batchesPerEpoch,'x');
    curves = Curves.process(curves,@(y) -y,'y');
    
    curves = Curves.process(curves,@(y) gather(y), 'y');
    if size(curves{1}.y,1)>size(curves{1}.y,2)
        curves = Curves.process(curves,@(y) mean(y,2), 'y');
    else
        curves = Curves.process(curves,@(y) mean(y,1)', 'y');
    end
    minLength = min(cell2mat(Curves.aggregate(curves,@(curve) length(curve.y))));
    curves = Curves.process(curves,@(y) y(1:minLength), 'y');
    curves = Curves.process(curves,@(x) x(1:minLength), 'x');
    
    sgd = curves{1};
    curves{4}.avgTime = sgd.avgTime*1.1;
    curves = Curves.process(curves,@(exp) Curves.normalizeX(exp,sgd.avgTime));
    opts.xlim = [0 max(sgd.x)];
    
    x = Curves.aggregate(curves,'x');
    y = Curves.aggregate(curves,'y');
    labels = Curves.aggregate(curves,'label');

	prettyPlotting(x,y,labels,opts);
    print('-dpdf',saveAs); 
    
    
end