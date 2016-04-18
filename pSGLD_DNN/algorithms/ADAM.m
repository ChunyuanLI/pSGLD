function [err,state] = ADAM(model,f,opts,state) %--TODO: Add momentum
%% error checking and initialization
    if nargin < 3; state = {}; end
    if ~isfield(state,'iter'); state.iter = 0; end
    state.iter = state.iter+1;

    
    if isfield(opts,'learningRate')
        lr = opts.learningRate;
    else
        lr = 1;
    end
    
    if isfield(opts,'RMSpropDecay')
        rmsd1 = opts.RMSpropDecay1;
        rmsd2 = opts.RMSpropDecay2;
    else
        rmsd1 = 0.9;
        rmsd2 = 0.999;
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
    
        if isfield(opts,'learningRateOffset') && opts.learningRateOffset >0
            lr=opts.learningRate*((state.iter+opts.learningRateOffset)^-opts.learningRateDecay);
            
        elseif isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
            lr= ...
                opts.learningRate*(state.iter^-opts.learningRateDecay);
            %         lr = lr/(opts.learningRateDecay^state.iter);
        end
    
    if ~isfield(state,'history1'); state.history1 = 0+grad; end
    if ~isfield(state,'history2'); state.history2 = 0+grad.^2; end
    
    %% algorithm
    state.history1 = rmsd1*state.history1 + (1-rmsd1)*grad;
    state.history2 = rmsd2*state.history2 + (1-rmsd2)*grad.^2;
    moment1 = state.history1/(1-rmsd1^state.iter);
    moment2 = state.history2/(1-rmsd2^state.iter);
    
    grad = moment1 ./ (eps + sqrt(moment2));

    params = params - lr*grad;

    model.setParameters(params);
end