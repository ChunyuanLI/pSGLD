function state =  SpectralDescentEdo(model,opts,state)    
    if isfield(opts,'learningRate')
        lr = opts.learningRate;
    else
        lr = 1;
    end

    term1 = 1;
    term2 = 1;
    l = 0;
    for ii=numel(model.modules):-1:1
        if isa(model.modules{ii},'Linear')
                l = l+1;
                [~,S,~] = svd(model.modules{ii}.weights);
                model.modules{ii}.weightsStepSize = lr*1 / ( (1/(4)^(3*l)) * numel(model.modules{ii}.weights)^2 * term1 * term2 );
                model.modules{ii}.accWeightGrads = projectMatrix(model.modules{ii}.accWeightGrads,opts.gpu,'spectraltrue');
                if ii>1
                    lsv = S(1,1);
                    term1 = term1 * lsv;
                    term2 = 1 + term2*lsv;
                end
                %fprintf('Module %d, stepSize=%g\n',ii,model.modules{ii}.weightsStepSize);
        end
    end
    model.updateParameters();
end        
