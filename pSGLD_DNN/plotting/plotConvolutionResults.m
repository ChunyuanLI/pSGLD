function ConvolutionResults()
    %% CIFAR
    files = {'exps/cifar-1L-CNN-LinSpectral.mat','cifar-1L-CNN-NonSpect.mat','cifar-1L-CNN-RMSprop.mat','cifar-1L-CNN-SSD.mat'};
    relabel = { {'GD','SGD'}  {'Adagrad','AdaGrad'}  {'RMSSpectral-a', 'RMSspectral'} {'AdaSpectral-a', 'AdaSpectral'} {'RMSprop-.01', 'RMSprop'}};
    select = {'SGD','AdaGrad','RMSprop','SSD','AdaSpectral','RMSspectral'};
    saveAs = 'exps/CIFAR-1Layer-CNN-trainErrors.pdf';
    batchesPerEpoch = 25;
    
    opts.xlabel = 'Normalized time';
    opts.ylabel = '$\log p(\mathbf{v})$';
    opts.title = 'Cifar, CNN';
    opts.fontSize = 25;
    opts.markerFrequency = 4;
    opts.yscale = 'log';
    
    structure = Curves.aggregate(files,@Curves.fileLoader); % load file contents
    structure = Curves.flatten(structure);
    
    accessor = getExpsAccessor('trainErrors');
	curves = Curves.aggregate(structure,accessor);
    
    curves = Curves.filter(curves, @(curve) ~strcmp(curve.label,'RMSprop'));
    curves = Curves.filter(curves, @(curve) ~Curves.endsWith(curve.label,'Bias'));
    
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

    
   
    
    
end