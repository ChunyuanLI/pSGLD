function state = AdaSpectral(model,opts,state) %--TODO: Add momentum

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
    eps = 1e-2;
end


params = model.getParameters();
grad = model.getParametersGradient();


if isfield(opts,'weightDecay') && opts.weightDecay > 0
    grad = grad + opts.weightDecay*params;
end

if isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
    lr = lr/(opts.learningRateDecay^state.iter);
end
%%
L=numel(model.modules)/2;
if ~isfield(state,'history');
    
    state.history = cell(L,1);
    for l=1:L
        siz=size(model.modules{2*l-1}.weights);
        if strcmp(class(model.modules{2*l-1}.weights),'gpuArray')
            state.history{l}=gpuArray.zeros(siz+[0 1]);
        else
        state.history{l}=zeros(siz+[0 1]);
        end
    end
end
%%
for l=1:L
    H = [model.modules{2*l-1}.accBiasGrads model.modules{2*l-1}.accWeightGrads];
    if sum(isnan(H(:)))>0
        1;
    end
    state.history{l} = state.history{l}+ H.^2;
    history=eps+sqrt(state.history{l});
    hist2=sqrt(history);
    k=min(min(size(history)),30);
%     try
    H = lr*projectMatrix(H./hist2,opts.gpu,'approxRandSpectral',k)./hist2;
%     catch
%         1;
%     end
    model.modules{2*l-1}.accBiasGrads=H(:,1);
    model.modules{2*l-1}.accWeightGrads=H(:,2:end);
    %     size(H)
end
model.updateParameters();
%%
%     state.history = state.history + grad.^2;
%     history=eps+sqrt(state.history);
%     hist2=sqrt(history);
% %     grad = grad ./ (eps + sqrt(state.history));
%     k=min(min(size(grad)),50);
%     grad = projectMatrix(grad./hist2,opts.gpu,'approxRandSpectral',k)./hist2;

%     H = [model.modules{ii}.accBiasGrads model.modules{ii}.accWeightGrads];
%     H = projectMatrix(H,opts.gpu,'approxRandSpectral',k);
%     if sum(isnan(H(:)))>0
%         1;
%     end

%     params = params - lr*grad;

%     model.setParameters(params);
end