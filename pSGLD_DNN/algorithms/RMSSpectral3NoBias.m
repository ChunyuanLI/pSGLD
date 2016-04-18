function [err,state] =RMSSpectral3(model,f,opts,state) %--TODO: Add momentum

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


err = f();

if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
    lr = lr*(state.iter^-opts.learningRateDecay);
end

if isfield(opts,'RMSpropDecay')
    rmsd = opts.RMSpropDecay;
else
    rmsd = 0.99;
end
%%
L=numel(model.modules);
if ~isfield(state,'history');
    
    state.history = cell(L,1);
    for l=1:L
        if isa((model.modules{l}),'Linear')
            siz=size(model.modules{l}.weights);
            if isa((model.modules{l}.weights),'gpuArray')
                state.history{l}=gpuArray.ones(siz);
            else
                state.history{l}=ones(siz);
            end
        elseif isa((model.modules{l}),'SpatialConvolution')
            state.history{l}=zeros(size(model.modules{l}.weights));
            if opts.gpu; state.history{l} = gpuArray(state.history{l}); end
        end
    end
end

%%
for l=1:L
    if isa((model.modules{l}),'Linear')
        H = model.modules{l}.accWeightGrads;
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            H = H + opts.weightDecay*model.modules{l}.weights;
            if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
        end
        if sum(isnan(H(:)))>0
            1;
        end
        state.history{l} = rmsd*state.history{l}+ (1-rmsd)*H.^2;
        history=eps+sqrt(state.history{l});
        hist2=sqrt(history);
        k=min(min(size(history)),30);
        %     try
        H = lr*projectMatrix(H./hist2,opts.gpu,'approxRandSpectral',k)./hist2;
        %     catch
        %         1;
        %     end
        model.modules{l}.accWeightGrads=H;
    elseif isa(model.modules{l},'SpatialConvolution') && (isfield(opts,'projectKernels') && opts.projectKernels)
        H = model.modules{l}.accWeightGrads;
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            H = H+ opts.weightDecay*model.modules{l}.weights;
            if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
        end
        state.history{l} = rmsd*state.history{l}+ (1-rmsd)*H.^2;
        history=eps+sqrt(state.history{l});
        hist2=sqrt(history);
        k=min(min(size(history)),30);
        H = lr*projectKernelsv2(H./hist2,opts.gpu,'approxRandSpectral',k)./hist2;
        model.modules{l}.accWeightGrads=H;
    end
end
model.updateParameters();
end