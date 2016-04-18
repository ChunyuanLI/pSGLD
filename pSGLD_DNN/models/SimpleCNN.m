function model = SimpleCNN(inSize,convSizes, linSizes, actFunc, gpu, intialParameters)
    if nargin < 5
        gpu = false;
    end
    opts.gpu = gpu;
    opts.precision = 'single';
    model = Sequential(opts);
    
    outSize = inSize(1:2);
    
    inChannels = inSize(3);
    for ii=1:size(convSizes,1)
        c = SpatialConvolution(inChannels,convSizes(ii,3),convSizes(ii,1),convSizes(ii,2),opts);
        % c.padding = [floor((convSizes(ii,1)-1)/2), floor((convSizes(ii,1)-1)/2), floor((convSizes(ii,2)-1)/2), floor((convSizes(ii,2)-1)/2)];
        c.padding = [4,4,4,4];
        p = Pooling(convSizes(ii,4), convSizes(ii,5));
        p.stride = [convSizes(ii,4), convSizes(ii,5)];
        outSize = p.getOutputSize(c.getOutputSize(outSize));
        inChannels = convSizes(ii,3);
        model.add(c);
        model.add(p);
        model.add(eval(actFunc));
    end
    model.add(Flatten());
    try
        [outSize convSizes(end,3)]
    linInSize = prod([outSize convSizes(end,3)])
    catch
        linInSize = prod(inSize(1:2));
    end
    model.add(Linear(linInSize,linSizes(1),opts));
    for ii=1:(numel(linSizes)-1)
        model.add(eval(actFunc));
        model.add(Linear(linSizes(ii),linSizes(ii+1),opts));
    end

    model.add(LogSoftmax());
    if nargin>=6
    model.setParameters(intialParameters);
    end
end