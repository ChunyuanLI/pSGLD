classdef StepSizeTrackingExperiment < SimpleExperiment
    properties
        stepSizeHistory = [];
    end
    methods
        function obj = StepSizeTrackingExperiment(name, modelFactory, dataFactory, optim)
            obj = obj@SimpleExperiment(name, modelFactory, dataFactory, optim);
        end
        
        function errs = trainEpoch(obj, model, data)
            criterion = NLLCriterion();
            dataSize = numel(data.labels);
            shuffle = randperm(dataSize);
            batch = 0;
            errs = [];
            model.setParametersGradient(zeros(model.parameterSize,1));
            for lo = 1:obj.descentOpts.batchSize:dataSize
                batch = batch+1;
                hi = min(lo+obj.descentOpts.batchSize-1,dataSize);
                inputs = data.inputs(:,shuffle(lo:hi));
                labels = data.labels(shuffle(lo:hi));
                preds = model.forward(inputs);
                err = criterion.forward(preds,labels)/(hi-lo+1);
                if obj.reportBatchInterval > 0 && mod(batch,obj.reportBatchInterval) == 0
                    fprintf('Batch error: %g\n',err);
                end
                errs = [errs err];
                grads = criterion.backward(preds,labels);
                grads = grads./(hi-lo+1); % average according to batch size
                
                model.backward(inputs,grads);
                obj.optimState = obj.optim(model,obj.descentOpts,obj.optimState); 
                stepSizes = [];
                for ii=1:numel(model.modules)
                    if isa(model.modules{ii},'Linear')
                        stepSizes = [stepSizes model.modules{ii}.weightsStepSize];
                    end
                end
                obj.stepSizeHistory = [obj.stepSizeHistory ; stepSizes];
            end
        end
        
        function finish(obj)
            figure('name',sprintf('%s step size history',obj.name));
            plot(obj.stepSizeHistory/obj.descentOpts.learningRate);
            legendNames = {};
            for ii=1:size(obj.stepSizeHistory,2)
                legendNames = [legendNames;sprintf('Layer #%d',ii)];
            end
            legend(legendNames);
        end
        
    end

    
end