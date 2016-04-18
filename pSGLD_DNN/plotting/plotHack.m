function plotHack(structure,attr)
	if nargin < 2; attr = 'trainErrors'; end
    
    prefix = 'results-';
    
    function curve = hackLoader(path)
        curve.y = Curves.fileLoader(path).(attr);
        curve.x = (1:numel(curve.y))';
        pos = strfind(path,prefix);
        curve.label = path(pos+length(prefix):end-4);
    end
    
	if numel(structure) > 0

        structure = Curves.aggregate(structure,@Curves.dirLister); %read directory contents
        structure = Curves.flatten(structure);
        
        %get relevant quanitities for plotting
		curves = Curves.aggregate(structure,@hackLoader);
        curves = Curves.filter(curves,@(curve) strfind(curve.label,'RMSSpectral')==1 );
        
		
		%pre-process before plotting
        curves = Curves.process(curves,@(y) gather(y), 'y');
        curves = Curves.process(curves,@(y) reshape(y',numel(y),1), 'y');
		minLength = min(cell2mat(Curves.aggregate(curves,@(curve) length(curve.y))));
		curves = Curves.process(curves,@(y) y(1:minLength), 'y');
        curves = Curves.process(curves,@(x) x(1:minLength), 'x');
		curves = Curves.process(curves, @(y) Curves.smooth(y,10), 'y');
        
        curves = Curves.sort(curves,'label');
        
        %curves = Curves.findReplace(curves,{{'AdaSpectral-K3','AdaSpectral'}});
        
        %curves = Curves.squeeze(curves);
        %curves = Curves.mean(curves);
		
		%plot
		x = Curves.aggregate(curves,'x');
		y = Curves.aggregate(curves,'y');
		labels = Curves.aggregate(curves,'label');
		
        opts.ylabel = attr;
        %opts.ylim = [0 3];
		prettyPlotting(x,y,labels,opts);
	end
end