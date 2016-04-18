function plotStuff(structure,varargin)

    % read arguments
    p = inputParser;
    addRequired(p,'structure');
    addOptional(p,'attr','trainErrors',@ischar);
    addOptional(p,'showMarkers','',@(s) strcmp(s,'showMarkers'));
    addOptional(p,'sort','',@(s) strcmp(s,'sort'));
    addParameter(p,'accessor',@getExpsAccessor,@(x) isa(x,'function_handle'));
    addParameter(p,'relabel',{},@iscell);
    addParameter(p,'select',{},@iscell);
    addParameter(p,'saveAs','',@ischar);
    addParameter(p,'remove',{},@iscell);
    addParameter(p,'colors',[],@isnumeric);
    addParameter(p,'xlim',[],@isnumeric);
    addParameter(p,'ylim',[],@isnumeric);
    addParameter(p,'xscale','',@ischar);
    addParameter(p,'yscale','',@ischar);
    addParameter(p,'xlabel','',@ischar);
    addParameter(p,'ylabel','',@ischar);
    addParameter(p,'title','',@ischar);
    addParameter(p,'normalizeXBy','',@ischar);
    addParameter(p,'processX',@(x) x,@(x) isa(x,'function_handle'));
    addParameter(p,'processY',@(x) x,@(x) isa(x,'function_handle'));
    addParameter(p,'lineWidth',1,@isnumeric);
    addParameter(p,'markerSize',8,@isnumeric);
    addParameter(p,'startsWith','',@ischar);
    addParameter(p,'endsWith','',@ischar);
    addParameter(p,'notStartsWith','',@ischar);
    addParameter(p,'notEndsWith','',@ischar);
    addParameter(p,'smoothingWindow',0,@isnumeric);
    parse(p,structure,varargin{:});
    
    opts.lineWidth = p.Results.lineWidth;
    opts.markerSize = p.Results.markerSize;
    if ~isempty(p.Results.colors); opts.colors = p.Results.colors; end
    if ~isempty(p.Results.xscale); opts.xscale = p.Results.xscale; end
    if ~isempty(p.Results.yscale); opts.yscale = p.Results.yscale; end
    if ~isempty(p.Results.xlabel); opts.xlabel = p.Results.xlabel; end
    if ~isempty(p.Results.ylabel); opts.ylabel = p.Results.ylabel; end
    if ~isempty(p.Results.showMarkers); opts.showMarkers = true; end
    if ~isempty(p.Results.title); opts.title = p.Results.title; end
    if ~isempty(p.Results.xlim); opts.xlim = p.Results.xlim; end
    if ~isempty(p.Results.ylim); opts.ylim = p.Results.ylim; end
    

    
	if numel(structure) > 0
		
		
        if isa(structure{1},'char') && exist(structure{1},'dir')==7
            structure = Curves.aggregate(structure,@Curves.dirLister); %read directory contents
            structure = Curves.flatten(structure);
        end
		
        
        if isa(structure{1},'char') && exist(structure{1},'file')==2
            structure = Curves.aggregate(structure,@Curves.fileLoader); % load file contents
            structure = Curves.flatten(structure);
        end
		
		
		%get relevant quanitities for plotting
		accessor = p.Results.accessor(p.Results.attr);
		curves = Curves.aggregate(structure,accessor);
        
        if numel(p.Results.relabel) > 0
            curves = Curves.findReplace(curves,'label',p.Results.relabel);
        end
        
        if ~isempty(p.Results.startsWith)
            curves = Curves.filter(curves,@(curve) Curves.startsWith(curve.label,p.Results.startsWith) );
        end
        
        if ~isempty(p.Results.endsWith)
            curves = Curves.filter(curves,@(curve) Curves.endsWith(curve.label,p.Results.endsWith) );
        end
        
        if ~isempty(p.Results.notStartsWith)
            curves = Curves.filter(curves,@(curve) (~Curves.startsWith(curve.label,p.Results.notStartsWith) ));
        end
        
        if ~isempty(p.Results.notEndsWith)
            curves = Curves.filter(curves,@(curve) ~(Curves.endsWith(curve.label,p.Results.notEndsWith) ));
        end
        
        if  numel(p.Results.remove) > 0
            curves = Curves.filter(curves,@(curve) ~any(cellfun(@(label) strcmp(curve.label,label),p.Results.remove)) );
        end
        
        if numel(p.Results.select) > 0
            curves = Curves.select(curves,'label',p.Results.select);
        end
        
        curves = Curves.process(curves,p.Results.processX,'x');
        curves = Curves.process(curves,p.Results.processY,'y');
		
		%pre-process before plotting
        curves = Curves.process(curves,@(y) gather(y), 'y');
        if size(curves{1}.y,1)>size(curves{1}.y,2)
            curves = Curves.process(curves,@(y) mean(y,2), 'y');
        else
            curves = Curves.process(curves,@(y) mean(y,1)', 'y');
        end
        %curves = Curves.process(curves,@(y) y(:), 'y');
		minLength = min(cell2mat(Curves.aggregate(curves,@(curve) length(curve.y))));
		curves = Curves.process(curves,@(y) y(1:minLength), 'y');
        curves = Curves.process(curves,@(x) x(1:minLength), 'x');
        
        if ~isempty(p.Results.normalizeXBy)
            by = Curves.filter(curves,@(curve) strcmp(curve.label,p.Results.normalizeXBy));
            by = by{1};
            curves = Curves.process(curves,@(exp) Curves.normalizeX(exp,by.avgTime));
            if isempty(p.Results.xlim); opts.xlim = [0 max(by.x)]; end
        end
        
        if p.Results.smoothingWindow > 0
            curves = Curves.process(curves, @(y) Curves.smooth(y,p.Results.smoothingWindow), 'y');
        end
        
        if ~isempty(p.Results.sort); curves = Curves.sort(curves,'label'); end
        %curves = Curves.findReplace(curves,{{'AdaSpectral-K3','AdaSpectral'}});
        
        %curves = Curves.squeeze(curves);
        %curves = Curves.mean(curves);
        
		%plot
		x = Curves.aggregate(curves,'x');
		y = Curves.aggregate(curves,'y');
		labels = Curves.aggregate(curves,'label');
        
		h = prettyPlotting(x,y,labels,opts);
        
        if ~isempty(p.Results.saveAs)
            print('-dpdf',p.Results.saveAs); 
        end
	end
end