classdef BatchExperiment < Experiment
    properties
        descentOpts;
        results = {};
        duration;
        saveInterval = -1;
        trainError=[];
        optim;
        optimState = {};
        reportBatchInterval = 50;
        reportEpochInterval = 1;
    end
    methods
        function obj = BatchExperiment(name, modelFactory, dataFactory, optim)
            obj = obj@Experiment(name, modelFactory, dataFactory);
            obj.descentOpts.momentum = 0;
            obj.descentOpts.learningRateDecay = 0;
            obj.descentOpts.L2WeightDecay = 0;
            obj.descentOpts.learningRate = 0.1;
            obj.descentOpts.gpu = false;
            
            obj.descentOpts.initialEpoch = 1;
            obj.descentOpts.epochs = 10;
            obj.descentOpts.batchSize = 100;
            
            obj.optim = optim;
        end
        
        
        function runExperiment(obj)
            disp(obj.name)
            trainErrors = [];
            tic;
            for epoch=obj.descentOpts.initialEpoch:obj.descentOpts.epochs
                %fprintf('Epoch %d...\n',epoch);
                rng(obj.randSeed+epoch,'twister');
                trainErrors = [trainErrors ; mean(obj.trainEpoch(obj.model,obj.data.train))];
                if obj.saveInterval > 0 && mod(epoch,obj.saveInterval) == 0
                    obj.save();
                end
                if obj.reportEpochInterval > 0 && mod(epoch,obj.reportEpochInterval) == 0
                    fprintf('Epoch error: %g\n',mean(trainErrors(end,:)));
                end
            end
            obj.duration = toc;
            obj.results.trainErrors = trainErrors;
            obj.results.trainAccuracy = obj.evaluate(obj.model,obj.data.train);
            obj.results.testAccuracy = obj.evaluate(obj.model,obj.data.test);
            disp(obj.name)
            fprintf('Training set accuracy: %g\n',obj.results.trainAccuracy);
            fprintf('Test set accuracy: %g\n',obj.results.testAccuracy);
        end
        
        
        function errs = trainEpoch(obj, model, data)
            criterion = NLLCriterion();
            dataSize = numel(data.labels);
            shuffle = randperm(dataSize);
            batch = 0;
            errs = [];
            
            model.setParametersGradient(zeros(model.parameterSize,1));
            hi=0;
            try
                obj.optimState.iter;
            catch
                obj.optimState.iter=0;
            end
            %             for lo = 1:obj.descentOpts.batchSize:dataSize
            while true
                lo=hi+1;
                if lo>dataSize
                    break
                end
                batch = batch+1;
                
                batchSize=floor((obj.descentOpts.batchGain)^(obj.optimState.iter)*obj.descentOpts.batchBase);
                hi = min(lo+batchSize-1,dataSize);
                inputs = data.inputs(:,shuffle(lo:hi));
                labels = data.labels(shuffle(lo:hi));
                preds = model.forward(inputs);
                err = criterion.forward(preds,labels)/(hi-lo+1);
                if obj.reportBatchInterval > 0 && mod(batch,obj.reportBatchInterval) == 0
                    fprintf('Batch error: %g\n',err);
                end
                errs = [errs err];
                grads = criterion.backward(preds,labels);
                grads = grads./(hi-lo+1)*(hi-lo+1)./batchSize; % average according to batch size
                
                model.backward(inputs,grads);
                obj.optimState = obj.optim(model,obj.descentOpts,obj.optimState);
            end
                                fprintf('Batch Size: %d\n',batchSize);
            1;
        end
        
        function acc = evaluate(obj,model,data)
            preds = model.forward(data.inputs);
            [~, maxInd] = max(preds);
            acc = sum(maxInd'==data.labels)/length(data.labels);
        end
        
        function sobj = saveobj(obj)
            sobj = saveobj@Experiment(obj);
            sobj.descentOpts = obj.descentOpts;
            sobj.optim = obj.optim;
            sobj.optimState = obj.optimState;
        end
        
        function obj = reload(obj,sobj)
            obj = reload@Experiment(obj,sobj);
            obj.descentOpts = sobj.descentOpts;
            obj.optim = sobj.optim;
            obj.optimState = sobj.optimState;
        end
        
    end
    methods (Static)
        function obj = loadobj(sobj)
            obj = SimpleExperiment(sobj.name,sobj.modelFactory,sobj.dataFactory,sobj.optim);
            obj.reload(sobj);
        end
    end
    
end