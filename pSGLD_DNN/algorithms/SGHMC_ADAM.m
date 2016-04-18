function [err,state] = SGHMC_ADAM(model,f,opts,state) %--TODO: Add momentum

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
        rmsd = opts.RMSpropDecay;
    else
        rmsd = 0.999;
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
    
    if ~isfield(state,'history'); state.history = 0+grad.^2; end
    
    % if ~isfield(state,'momentum'); state.momentum = randn(size(grad)); end
    if ~isfield(state,'momentum') || mod(state.iter,  opts.N) == 500; state.momentum = randn(size(grad)); end    
	if ~isfield(opts,'D')
		opts.D = 100 * sqrt(lr);
	end
	if ~isfield(state,'alpha')
		state.alpha = opts.D * ones(size(grad));
	end

    %% algorithm
    p = 4;
    state.history1 = rmsd1*state.history1 + (1-rmsd1)*grad;
    moment1 = state.history1/(1-rmsd1^state.iter);
    state.history2 = rmsd2*state.history2 + (1-rmsd2)*abs(grad).^(p/2);
    moment2 = state.history2/(1-rmsd2^( (p/2) *state.iter));    
    pcder= 1./ (eps + moment2.^(2/p));

    
   
    % grad = lr* moment1 .* pcder + sqrt(2*lr.*pcder).*randn(size(grad))/opts.N ;    
        
	%if state.iter < 5000
	%	pcder = 1;
	%else
    %pcder=(eps + sqrt(state.history));
	%end
	params = params + state.u ./ pcder;
	state.u = (1 - state.alpha) .* state.u - moment1 * lr ./ pcder + sqrt(2 * lr * opts.D) .* randn(size(grad))/opts.N ;
    state.momentum   = state.momentum - state.thermostat * lr .* state.momentum - opts.N*grad* lr;
    state.momentum   = state.momentum + sqrt(2 * opts.C * lr) * randn(size(state.momentum));
    
    model.setParameters(params);
end
