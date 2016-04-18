function state = SpectralDescentDavid(model,opts,state)
	lips = 1;
	L= numel(model.modules)/2;
	l = numel(model.modules)/2+1;
	%             L=numel(model.modules);z
	S=[];
	J=[];

	for ii=numel(model.modules):-1:1
		if isa(model.modules{ii},'Linear')
		    l = l-1;
		    tmp=svd(model.modules{ii}.weights);
		    S=[tmp(1),S];
		    if ii>1
		        J=[mean(sum(model.modules{ii-1}.output.^2)),J];
		    else
		        J=[mean(sum(opts.data.train.inputs.^2)),J];
		    end
		    %
		    if true
		        if l==L
		            model.modules{ii}.weightsStepSize=opts.learningRate*1/J(end);
		        else
		            model.modules{ii}.weightsStepSize=opts.learningRate*1./(J(1)*(.5*S(2)+.25*S(2)^2));
		        end
		    else
		        if l==L
		            model.modules{ii}.weightsStepSize=1/J(end);
		            steps(l)=1/J(end);
		            N1s(l)=0;
		            N2s(l)=J(end);
		        else
		            %%
		            omega=cumprod(S(2:end));
		            %%
		            N2=J(1)*omega(end)^2;
		            N1=.5*(J(1))*omega(end)^2;%*(1+sum(omega));
		            N1s(l)=N1;
		            N2s(l)=N2;
		            Lip=(4*N1+N2);
		            stepsize=1./Lip;
		            %%
		            model.modules{ii}.weightsStepSize=stepsize;
		            steps(l)=stepsize;
		        end
		    end

			if isfield(opts,'weightDecay') && opts.weightDecay > 0
                model.modules{ii}.accWeightGrads = model.modules{ii}.accWeightGrads + opts.weightDecay*model.modules{ii}.weights;
			end

		    trueProj=true;
		    if trueProj
		        H = projectMatrix(model.modules{ii}.accWeightGrads,opts.gpu,'spectraltrue');
		    else
		        H = projectMatrix(model.modules{ii}.accWeightGrads,opts.gpu,'approxSpectral');
		    end
		    if sum(isnan(H(:)))>0
		        1;
		    end
		    model.modules{ii}.accWeightGrads=H;
		end
	end
    model.updateParameters();
end