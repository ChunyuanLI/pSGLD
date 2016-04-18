function model = SimpleNN(sizes, actFunc, gpu,intialParameters)
    if nargin < 3
        gpu = false;
    end
    opts.gpu = gpu;
    model = Sequential(opts);
    model.add(Linear(sizes(1),sizes(2),opts));
    for ii=2:(numel(sizes)-1)
        model.add(eval(actFunc));
        model.add(Linear(sizes(ii),sizes(ii+1),opts));
    end
    model.add(LogSoftmax());
    if nargin>=4
    model.setParameters(intialParameters);
    end
end