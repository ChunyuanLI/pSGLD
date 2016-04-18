function plotOverfitting()
    %% CIFAR
    files = {'exps/cifar-FFNN-VarSize-Rest.mat','cifar-FFNN-VarSize-RMSspectral.mat',};
    saveAs = 'exps/Cifar10-FFNN-VarSize-trainAccuracy.pdf';
    
    opts.xlabel = 'Hidden layer size';
    opts.ylabel = 'Accureacy';
    opts.title = 'Cifar-10, 2-Layer NN';
    opts.fontSize = 25;
    opts.markerFrequency = 1;
    opts.ylim = [0 1];
    opts.xlim = [0 50];
    
    structure = Curves.aggregate(files,@Curves.fileLoader); % load file contents
    structure = Curves.flatten(structure);
    
    accessor = getExpsAccessor('trainAccuracy');
	curves = Curves.aggregate(structure,accessor);
    
    curves = Curves.sort(curves,'label');
    curves = Curves.process(curves, @clipLabel,'label');
    curves = Curves.squeeze(curves,'label',@concatExps);
    
    curves = Curves.process(curves,@(y) gather(y), 'y');
    if size(curves{1}.y,1)>size(curves{1}.y,2)
        curves = Curves.process(curves,@(y) mean(y,2), 'y');
    else
        curves = Curves.process(curves,@(y) mean(y,1)', 'y');
    end
    minLength = min(cell2mat(Curves.aggregate(curves,@(curve) length(curve.y))));
    curves = Curves.process(curves,@(y) y(1:minLength), 'y');
    curves = Curves.process(curves,@(x) (1:minLength)*5, 'x');
    

    x = Curves.aggregate(curves,'x');
    y = Curves.aggregate(curves,'y');
    labels = Curves.aggregate(curves,'label');

	prettyPlotting(x,y,labels,opts);
    print('-dpdf',saveAs); 
    
    function a = concatExps(a,b)
        a.y = [a.y ; b.y];
    end

    function label = clipLabel(label)
        pos = strfind(label,'-');
        label = label(1:pos-1);
    end
    
end