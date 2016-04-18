function [err,state] = approxRandSpectralDescentDavid(model,f,opts,state)
k=30;
if nargin < 3; state = {}; end
if ~isfield(state,'iter'); state.iter = 0; end
state.iter = state.iter+1;
lips = 1;
L= numel(model.modules)/2;
l = numel(model.modules)/2+1;
%             L=numel(model.modules);z
f();
maxStep=10;
S=[];
G=[];
Gi=[];
Si=[];
J=[];
if isa(model.modules{2},'Sigmoid');
    q=.5;
    qt=1/(6*sqrt(3));
elseif isa(model.modules{2},'SReLU');
    q=1;
    qt=.5;
else isa(model.modules{2},'SReLU');
    q=1;
    qt=.5;
end
for ii=numel(model.modules):-1:1
    if isa(model.modules{ii},'Linear')
        l = l-1;
        tmp=powermethod(model.modules{ii}.weights,1e-1);
        %         tmp=svd(model.modules{ii}.weights);
        S=[tmp(1),S];

        T=model.modules{ii}.gradInput.^2;
                N=size(T,2);
        G=[sqrt(mean(sum(T)))*(N),G];
        Gi=[sqrt(mean(max(T)))*(N),Gi];
        Si=[max(abs(model.modules{ii}.weights(:))),Si];
        if ii>1
            J=[mean(sum(model.modules{ii-1}.output.^2)),J];
        else
            J=[mean(sum(opts.data.train.inputs(:,1:100).^2)),J];
            %             J=[mean(sum(opts.data.train.inputs.^2)),J];
        end
        %
        
        if l==L
            
            stepsize=1/J(end);
            stepsize=min(maxStep,stepsize);
            steps(l)=stepsize;
            model.modules{ii}.weightsStepSize=stepsize;
        else
            N0=J(1)*G(2);
            Ss=cumprod(S(2:end-1));
            Ss=[1,Ss];
%             Ssi=Gi(2:end).*Ss;
            Ssi=Gi(2:end).*Ss;
            Ssi=q.^(1:numel(Ssi)).*Ssi;
            N1=.5*qt.*sum(Ssi);
            stepsize=1./(N0+4*N1);
            stepsize=min(maxStep,stepsize);
            steps(l)=stepsize;
            model.modules{ii}.weightsStepSize=stepsize;
        end
        
        if isfield(opts,'weightDecay') && opts.weightDecay > 0
            model.modules{ii}.accWeightGrads = model.modules{ii}.accWeightGrads + opts.weightDecay*model.modules{ii}.weights;
        end
        1;
        if isfield(opts,'learningRate') && opts.learningRate > 0
            model.modules{ii}.weightsStepSize = ...
                model.modules{ii}.weightsStepSize*opts.learningRate;
            %         lr = lr/(opts.learningRateDecay^state.iter);
        end
        
        if isfield(opts,'learningRateOffset') && opts.learningRateOffset >0
            model.modules{ii}.weightsStepSize*((state.iter+opts.learningRateOffset)^-opts.learningRateDecay);
            
        elseif isfield(opts,'learningRateDecay') && opts.learningRateDecay > 0
            model.modules{ii}.weightsStepSize = ...
                model.modules{ii}.weightsStepSize*(state.iter^-opts.learningRateDecay);
            %         lr = lr/(opts.learningRateDecay^state.iter);
        end
        
        H = [model.modules{ii}.accBiasGrads model.modules{ii}.accWeightGrads];
        H = projectMatrix(H,opts.gpu,'approxRandSpectral',k);
        if sum(isnan(H(:)))>0
            1;
        end
        model.modules{ii}.accBiasGrads=H(:,1);
        model.modules{ii}.accWeightGrads=H(:,2:end);
    end
end
model.updateParameters();
end