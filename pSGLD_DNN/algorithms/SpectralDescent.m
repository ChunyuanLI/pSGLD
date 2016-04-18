function [err,state] =SpectralDescent(model,f,opts,state) %--TODO: Add momentum

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

err = f();


if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
    lr = lr*(state.iter^-opts.learningRateDecay);
end


%%
L=numel(model.modules);

%%
for l=1:L
    if isa(class(model.modules{l}),'Linear')
        H = [model.modules{l}.accBiasGrads model.modules{l}.accWeightGrads];
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            H(:,2:end) = H(:,2:end)+ opts.weightDecay*model.modules{l}.weights;
            if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
        end
        if sum(isnan(H(:)))>0
            1;
        end

        k=min(min(size(model.modules{l}.weights)),30);
        H = lr*projectMatrix(H,opts.gpu,'approxRandSpectral',k);

        model.modules{l}.accBiasGrads=H(:,1);
        model.modules{l}.accWeightGrads=H(:,2:end);
    
    elseif isa(model.modules{l},'SpatialConvolution') && (isfield(opts,'projectKernels') && opts.projectKernels)
        H = model.modules{l}.accWeightGrads;
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            H = H+ opts.weightDecay*model.modules{l}.weights;
            if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
        end
        
        k=min(min(size(model.modules{l}.weights)),30);
        H = lr*projectKernelsv2(H,opts.gpu,'approxRandSpectral',k);
        model.modules{l}.accWeightGrads=H;
    end

end
model.updateParameters();

end