function [err,state] = SGNHTE_RMSprop(model,f,opts,state) %--TODO: Add momentum
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
    
    if ~isfield(state,'momentum'); state.momentum = 0; end
    if ~isfield(state,'thermostat'); state.thermostat = 0; end
    if ~isfield(state,'history'); state.history = 0+grad.^2; end
    
    %% algorithm
%     params     = params + state.momentum * lr;
%     state.thermostat = state.thermostat + (state.momentum.*state.momentum-1) * lr;
%     state.momentum   = state.thermostat * lr .* state.momentum - opts.N*grad* lr;
%     state.momentum   = state.momentum + sqrt(2 * opts.C * lr) * randn(size(state.momentum));   

    state.history = rmsd2*state.history + (1-rmsd2)*grad.^2;
    pcder=1./(eps + sqrt(state.history));   pcder_ = sqrt(pcder);
    
    grad = lr* grad ./ pcder + sqrt(2*lr./pcder).*randn(size(grad))/opts.N ;
    
    params     = params + pcder_ .* state.momentum * lr;
    state.thermostat = state.thermostat + (state.momentum.*state.momentum-1) * lr;
    state.momentum   = state.momentum - pcder .* state.thermostat * lr .* state.momentum - opts.N* pcder_ .* grad* lr;
    state.momentum   = state.momentum + sqrt( (2*pcder-lr*opts.C) * lr) .* randn(size(state.momentum));

    model.setParameters(params);
end