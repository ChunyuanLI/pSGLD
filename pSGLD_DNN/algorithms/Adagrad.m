function [err,state] = Adagrad(model,f,opts,state) %--TODO: Add momentum

    if nargin < 3; state = {}; end
    if ~isfield(state,'iter'); state.iter = 0; end
    state.iter = state.iter+1;
    
    if isfield(opts,'learningRate')
        lr = opts.learningRate;
    else
        lr = 1;
    end
    
    if isfield(opts,'epsilon')
        eps = opts.epsilon;
    else
        eps = 1e-1;
    end
    
    if ~isfield(opts,'reportL2Penalty');opts.reportL2Penalty=false;end

    params = model.getParameters();
    [err, grad] = f();
   
    
    if isfield(opts,'weightDecay') && opts.weightDecay > 0
        grad = grad + opts.weightDecay*params;
        if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(params(:),params(:)); end
    end
    
    if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
        lr = lr*(state.iter^-opts.learningRateDecay);
    end
    
    if ~isfield(state,'history'); state.history = zeros(model.parameterSize,1); end
    state.history = state.history + grad.^2;
    
    grad = grad ./ (eps + sqrt(state.history));
    
    params = params - lr*grad;

    model.setParameters(params);
end