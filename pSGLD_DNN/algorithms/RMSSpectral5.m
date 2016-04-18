function [err,state] =RMSSpectral5(model,f,opts,state) %--TODO: Add momentum

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
        if ~isreal(err)
            keyboard;
        end

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
    state.biasHistory = cell(L,1);
    for l=1:L
        if isa((model.modules{l}),'Linear')
            siz=size(model.modules{l}.weights);
            if isa((model.modules{l}.weights),'gpuArray')
                state.history{l}=gpuArray.ones(siz);
                state.biasHistory{l}=gpuArray.ones(siz(1),1);
%                 state.biasHistroy{l}=gpuArray,oness
            else
                state.history{l}=ones(siz);
                state.biasHistory{l}=ones(siz(1),1);
            end
        elseif isa((model.modules{l}),'SpatialConvolution')
            state.history{l}=ones(size(model.modules{l}.weights));
            if opts.gpu; state.history{l} = gpuArray(state.history{l}); end
        end
    end
end

%%
for l=1:L
    if isa((model.modules{l}),'Linear')
        H = [model.modules{l}.accBiasGrads model.modules{l}.accWeightGrads];
        bias=H(:,1);
        matr=H(:,2:end);
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            matr = matr+ opts.weightDecay*model.modules{l}.weights;
%             if opts.reportL2Penalty; err = err + 0.5*opts.weightDecay*dot(model.modules{l}.weights(:),model.modules{l}.weights(:)); end
        end
        if sum(isnan(H(:)))>0
            1;
        end
        state.history{l} = rmsd*state.history{l}+ (1-rmsd)*matr.^2;
        state.biasHistory{l} = rmsd*state.biasHistory{l} + (1-rmsd)*bias.^2;
        history=eps+sqrt(state.history{l});
        biasHistory=eps+sqrt(state.biasHistory{l});
        hist2=sqrt(history);
        k=min(min(size(history)),30);
        %     try
        matr = lr*projectMatrix(matr./hist2,opts.gpu,'approxRandSpectral',k)./hist2;
        bias = lr*bias./biasHistory;
        if ~isreal(bias) || ~isreal(matr)
            keyboard
        end
        %     catch
        %         1;
        %     end
        
        model.modules{l}.accBiasGrads=bias;
        model.modules{l}.accWeightGrads=matr;
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