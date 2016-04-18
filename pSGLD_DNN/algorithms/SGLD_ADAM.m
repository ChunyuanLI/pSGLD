function [err,state] = SGLD_ADAM(model,f,opts,state) %--TODO: Add momentum
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
        rmsd2 = 0.9999;
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
    if ~isfield(opts,'norm'); opts.norm = 4; end
    p = opts.norm;
    
    if ~isfield(state,'history1'); state.history1 = grad; end
  %   if ~isfield(state,'history2'); state.history2 = 0; end
    if ~isfield(state,'history2'); state.history2 = 0+abs(grad).^(p/2); end
    
    %% algorithm
%     state.history1 = (state.iter-1)*state.history1 + grad;
%     state.history1 = state.history1/state.iter;
   state.history1 = rmsd1*state.history1 + (1-rmsd1)*grad;
 %  moment1 = state.history1/(1-rmsd1^state.iter);
   moment1 = state.history1;
%   moment1 = grad;   
    
%     state.history2 = rmsd2*state.history2 + (1-rmsd2)*grad.^2;
%     moment2 = state.history2/(1-rmsd2^state.iter);
%     pcder= 1./ (eps + sqrt(moment2));
%    state.history2 = rmsd2^(p/2)*state.history2 + (1-rmsd2^(p/2))*abs(grad).^(p/2);
    state.history2 = rmsd2*state.history2 + (1-rmsd2)*abs(grad).^(p/2);
    moment2 = state.history2/(1-rmsd2^((2/2) * state.iter));   
%    moment2 = state.history2;
    pcder= 1./ (eps + moment2.^(2/p));
    
%   lrt = (lr* (1-rmsd2^( (p/2) *state.iter)).^(2/p))/(1-rmsd1^state.iter);
%   lrt = (lr* (1-rmsd2^( 1 *state.iter)).^(2/p))/(1-rmsd1^state.iter);
%   grad = lr* moment1 .* pcder + sqrt(2* (1-rmsd1) .*pcder/(lrt*opts.N)).*randn(size(grad)) ;    
    
%    grad = lrt* moment1 .* pcder + sqrt(2*lrt.*pcder).*randn(size(grad))/opts.N ; 
    grad = lr* moment1 .* pcder + sqrt(2*lr.*pcder).*randn(size(grad))/opts.N ;    
    
    params = params - grad;

    model.setParameters(params);
end