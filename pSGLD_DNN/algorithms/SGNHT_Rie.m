function [err,state] = SGNHT_Rie(model,f,opts,state) %--TODO: Add momentum

    %% error checking and initialization
    if nargin < 3; state = {}; end
    if ~isfield(state,'iter'); state.iter = 0; end
    state.iter = state.iter+1;

    
    if isfield(opts,'learningRate')
	%if state.iter < opts.burnin
        %        lr = opts.learningRate/(1+state.iter)^0.1;
        %else
                lr = opts.learningRate;
        %end
    else
        lr = 1;
    end
    if isfield(opts,'RMSpropDecay')
        rmsd = opts.RMSpropDecay;
    else
        rmsd = 0.999;
    end

	if ~isfield(opts, 'decay_grad')
		opts.decay_grad = 0.1;
	end
        if ~isfield(opts, 'anne_rate')
                opts.anne_rate = 0.5;
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
%	if state.iter == opts.burnin
%                state.history = grad.^2;
%        end

	if ~isfield(state,'u')
		state.u = randn(size(grad)) * sqrt(lr);
	end
	if ~isfield(opts,'D')
		opts.D = 1000 * sqrt(lr); %% no precondition: 100
	end
	if ~isfield(state,'alpha')
		state.alpha = opts.D * ones(size(grad));
	end
%	if mod(state.iter, 600) == 0; fprintf('u norm = %f, xi norm = %f\n', norm(state.u), norm(state.alpha));end
	if ~isfield(state,'grad') %%% use this: 98.61, not: 98.62 (opt)
		state.grad = grad;
	end
	grad = opts.decay_grad * state.grad + (1 - opts.decay_grad) * grad;
	state.grad = grad;

    %% algorithm
	if state.iter < opts.burnin
		his = (1e-4 + sqrt(state.history)).^(0.5);
	else
		his = (eps + sqrt(state.history)).^(0.5);
	end
    state.history = rmsd*state.history + (1-rmsd)*grad.^2;
	%if state.iter < 5000
	%	pcder = 1;
	%else
	if state.iter < opts.burnin
		pcder=(1e-4 + sqrt(state.history)).^(0.5);
	else
    	pcder=(eps + sqrt(state.history)).^(0.5);
	end
	if state.iter < opts.burnin
		tmp = (1 - pcder./his);
	else
		tmp = 0;
	end
	%end
	params = params + state.u ./ pcder / 2; %assert(isfinite(sum(params(:))));
	state.alpha = state.alpha + (state.u.^2 - lr) / 2; %assert(isfinite(sum(state.alpha)));
	state.u = exp(-state.alpha/2) .* state.u - opts.N * grad * lr ./ pcder / 2;
	state.u = state.u + sqrt(2 * lr^1.5 ./ pcder ) .* randn(size(grad));
	%assert(isfinite(sum(state.u)));
	state.u = exp(-state.alpha/2) .* state.u - opts.N * grad * lr ./ pcder / 2;
	%assert(isfinite(sum(state.u)));
	state.alpha = state.alpha + (state.u.^2 - lr ) / 2; %assert(isfinite(sum(state.alpha)));
	params = params + state.u ./ pcder / 2; %assert(isfinite(sum(params)));
    
    model.setParameters(params);
end
