function h = prettyPlotting(X,Y,legendNames,opts)
    
    if nargin < 4
        opts = struct();
    end
    
    if ~isfield(opts,'markerFrequency'); opts.markerFrequency = 0; end
    if ~isfield(opts,'lineWidth'); opts.lineWidth = 1; end
    if ~isfield(opts,'lineStyle'); opts.lineStyle = '-'; end
    if ~isfield(opts,'MarkerSize'); opts.MarkerSize = 8; end
    if ~isfield(opts,'FontSize'); opts.FontSize = 10; end
    if ~isfield(opts,'legendPosition'); opts.legendPosition = 'best'; end
    
    N = numel(Y);
    
    markers = ['o','x','*','s','v','^','<','>','p','h','.','+'];
    markers=repmat(markers,1,ceil(numel(Y)/numel(markers)));

    h=figure;
    set(h,'defaultTextInterpreter','latex')
    hold on;
    


    %algs = keys(curves);
    if isfield(opts,'colors') && numel(opts.colors)>0
        colors = opts.colors;
    else
        colors = distinguishable_colors(numel(Y));
    end
    lines = [];
    for ii=1:N
        y = Y{ii};
       
        if numel(X) == N
            x = X{ii};
        else
            x = X;
        end
        
        plotOpts = {};
        if opts.markerFrequency > 0
            xmarkers = x(1:opts.markerFrequency:end); % place markers at these x-values
            ymarkers = y(1:opts.markerFrequency:end);
            plotOpts = [plotOpts xmarkers ymarkers markers(ii), nan, nan, [opts.lineStyle,markers(ii)]]; 
        end
        plotOpts = [plotOpts 'color' colors(ii,:)];
        plotOpts = [plotOpts 'lineWidth' opts.lineWidth];
        plotOpts = [plotOpts 'MarkerSize' opts.MarkerSize];
        
        if isfield(opts,'showMarkers') && opts.showMarkers
            twins = plot(x,y,opts.lineStyle,plotOpts{:});
            lines = [lines twins(3)];
        else
            lines(:,end+1) = plot(x,y,opts.lineStyle,plotOpts{:});
        end
    end
    
    box on;
    axis tight;
    orient landscape;
    
    ca = get(h,'CurrentAxes');
    
    if isfield(opts,'xscale') && ~isempty(opts.xscale); set(ca,'xscale',opts.xscale); end
    if isfield(opts,'yscale') && ~isempty(opts.yscale); set(ca,'yscale',opts.yscale); end
    
    if isfield(opts,'xlim') && ~isempty(opts.xlim); xlim(opts.xlim); end
    if isfield(opts,'ylim') && ~isempty(opts.ylim); ylim(opts.ylim); end
    
    set(ca,'FontSize',opts.FontSize);
    leg = legend(lines(1,:),legendNames{:},'Location',opts.legendPosition);
    set(leg,'FontSize',opts.FontSize);
    
    if isfield(opts,'xlabel') && ~isempty(opts.xlabel); xlabel(opts.xlabel,'Interpreter','latex','FontSize',opts.FontSize); end
    if isfield(opts,'ylabel') && ~isempty(opts.ylabel); ylabel(opts.ylabel,'Interpreter','latex','FontSize',opts.FontSize); end
    if isfield(opts,'title') && ~isempty(opts.title); title(opts.title,'Interpreter','latex','FontSize',opts.FontSize); end
    
    hold off;
end