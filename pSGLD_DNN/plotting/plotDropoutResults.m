function plotDropoutResults()
    %% MNIST
    files = {'dropout_pSAGLD/pSAGLD_Dropout_cell.mat'};
    relabel = {'pSAGLD 400', 'Adam 400', ...
    'pSAGLD 800', 'Adam 800', ...
    'pSAGLD 1200', 'Adam 1200'};
    % relabel = { {'GD','SGD'}  {'Adagrad','AdaGrad'}  {'RMSSpectral', 'RMSspectral'}};
    select = {'pSAGLD 400', 'Adam 400', ...
    'pSAGLD 800', 'Adam 800', ...
    'pSAGLD 1200', 'Adam 1200'};
    saveAs = 'exps/Dropout-trainErrors.pdf';
    batchesPerEpoch = 30;
    
    opts.xlabel = 'Normalized time';
    opts.ylabel = '$\log p(\mathbf{v})$';
    opts.title = 'MNIST, 3-Layer NN';
    opts.fontSize = 25;
    opts.markerFrequency = 4;
    % opts.yscale = 'log';
    
    
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
    
    
end