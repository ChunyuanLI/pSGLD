classdef SimpleExperiment < Experiment
    properties  
        descentOpts;
        results = {};
        duration;
        saveInterval = -1;
        trainError=[];
        optim;
        optimState = {};
        reportBatchInterval = 100;
        reportEpochInterval = 1;
        dataChunkSize = 10000;
        ave_preds = [];
    end
    methods
        function obj = SimpleExperiment(name, modelFactory, data, optim)
            obj = obj@Experiment(name, modelFactory, data);
            obj.descentOpts.momentum = 0;
            obj.descentOpts.learningRateDecay = 0;
            obj.descentOpts.L2WeightDecay = 0;
            obj.descentOpts.learningRate = 0.1;
            obj.descentOpts.gpu = false;
            obj.descentOpts.initialEpoch = 1;
            obj.descentOpts.epochs = 10;
            obj.descentOpts.batchSize = 100;        
            obj.descentOpts.iter = 0;    
            obj.descentOpts.learningRate_acc = 0;    
            obj.optim = optim;
        end
        
        
        function runExperiment(obj)
            disp(obj.name)
            fprintf('batchsize: %g, stepsize %g\n',obj.descentOpts.batchSize, obj.descentOpts.learningRate);
            fprintf('weightDecay: %g \n',obj.descentOpts.weightDecay*obj.descentOpts.N);
            obj.ave_preds = zeros(obj.data.outSize, obj.data.testSizeMax);
            trainErrors = [];
            trainAccuracy = [];
            testAccuracy = [];
            times=[];
            
            tic;
            for epoch=obj.descentOpts.initialEpoch:obj.descentOpts.epochs
                rng(obj.randSeed+epoch,'twister');
                
                [train_error, test_acc]= obj.trainEpoch(obj.model, ...
                    obj.data.train.getIterator(obj.dataChunkSize),...
                    obj.data.test.getIterator(obj.dataChunkSize));
                
                trainErrors = [trainErrors   train_error'];
                testAccuracy = [testAccuracy ; test_acc];

                times=[times;toc];
                if obj.saveInterval > 0 && mod(epoch,obj.saveInterval) == 0
                    obj.save();
                end
                if obj.reportEpochInterval > 0 && mod(epoch,obj.reportEpochInterval) == 0
                    % fprintf('Epoch %d error: %g\n',epoch,mean(trainErrors(:,end)));
                    fprintf('#iter %d, lr: %f, lr_c: %f \n', obj.descentOpts.iter, obj.descentOpts.learningRate, obj.descentOpts.learningRate_acc);
                    fprintf('Test set accuracy: %g\n',testAccuracy(end));
                end
            end
            obj.duration = toc;
            obj.results.trainErrors   = trainErrors;
            obj.results.testAccuracy  = testAccuracy;
            obj.results.times = times;
         
            disp(obj.name)
            fprintf('batchsize: %g, stepsize%g\n',obj.descentOpts.batchSize, obj.descentOpts.learningRate);
            fprintf('Test set accuracy: %g\n',obj.results.testAccuracy(end));
        end

        
        function [errs acc]= trainEpoch(obj, model, data, testdata)
            criterion = NLLCriterion();
            errs = [];
            batch = {};
            batch.number = 0;
            
            function [fx,dfdx] = f(x)
                if nargin>1 && ~isempty(x); model.setParameters(x);end
                
                preds = model.forward(batch.inputs);
                fx = criterion.forward(preds,batch.labels)/(hi-lo+1);
                if obj.reportBatchInterval > 0 && mod(batch.number,obj.reportBatchInterval) == 0
                   % fprintf('#iter %d, lr: %f\n', obj.descentOpts.iter, obj.descentOpts.learningRate);
                end
                
                grads = criterion.backward(preds,batch.labels);
                grads = grads./(hi-lo+1); % average according to batch size
                model.backward(batch.inputs,grads);
                
                if nargout > 1; dfdx = model.getParametersGradient(); end
            end
            
            

           
            dataChunk = data();  testdataChunk = testdata();
            prevErr=inf;
            while dataChunk.size > 0
                
                %fprintf('Chunk %d, from %d to %d -> %d samples.\n',dataChunk.ind,dataChunk.lo,dataChunk.hi,dataChunk.size);
                ind(1:ndims(dataChunk.inputs)) = {':'};
                for lo = 1:obj.descentOpts.batchSize:dataChunk.size
                    
                    % train
                    obj.descentOpts.iter =  obj.descentOpts.iter + 1; 
                    if isfield(obj.descentOpts,'learningRateBlockDecay') && obj.descentOpts.learningRateBlockDecay > 0 ...
                            && mod(obj.descentOpts.iter, obj.descentOpts.learningRateBlock) == 0
                        obj.descentOpts.learningRate= obj.descentOpts.learningRate*obj.descentOpts.learningRateBlockDecay; 
                    end
                    
                    batch.number = batch.number+1;
                    hi = min(lo+obj.descentOpts.batchSize-1,dataChunk.size);
                    ind{end} = lo:hi;%shuffle(lo:hi);
                    batch.inputs = dataChunk.inputs(ind{:});
                    batch.labels = dataChunk.labels(lo:hi);
                    model.setParametersGradient(zeros(model.parameterSize,1));
                    [err, obj.optimState] = obj.optim(obj.model,@f,obj.descentOpts,obj.optimState);
                    if err>prevErr-.01;
                        keyboard
                    end
                    errs = [errs err]; 
             
                    % test
                    
                    if obj.descentOpts.iter > obj.descentOpts.burnin  && mod(obj.descentOpts.iter, 300)== 0
                        if strcmp(obj.name,'RMSprop')||  strcmp(obj.name,'SGLD\_RMSprop') || strcmp(obj.name,'SGLD') || strcmp(obj.name,'SGLD\_ADAM') ||...
                                strcmp(obj.name,'SGNHT\_S') || strcmp(obj.name,'SGNHT\_E') || strcmp(obj.name,'SGHMC\_E')...
                                || strcmp(obj.name,'SGNHT\_Rie') ...
                                || strcmp(obj.name,'SGNHTE\_RMSprop') || strcmp(obj.name,'SGLD\_ADAMmax'); 
%                             preds_acc = model.forward(testdataChunk.inputs)* obj.descentOpts.learningRate + obj.ave_preds*obj.descentOpts.learningRate_acc;
%                             obj.descentOpts.learningRate_acc  = obj.descentOpts.learningRate_acc + obj.descentOpts.learningRate;
%                             preds     = preds_acc/obj.descentOpts.learningRate_acc;
%                             obj.ave_preds = preds;
                            preds_acc = model.forward(testdataChunk.inputs)*obj.descentOpts.learningRate  + obj.ave_preds* obj.descentOpts.learningRate_acc;
                            obj.descentOpts.learningRate_acc  = obj.descentOpts.learningRate_acc + 1;
                            preds     = preds_acc/obj.descentOpts.learningRate_acc;
                            obj.ave_preds = preds;                            
                        elseif strcmp(obj.name,'SGD') || strcmp(obj.name,'ADAM') || strcmp(obj.name,'SSD')...
                                || strcmp(obj.name,'SSD-K')
                            preds = model.forward(testdataChunk.inputs);
                        end
                        [~, maxInd] = max(preds);
                        acc = 0; 
                        acc = acc+sum(maxInd'==testdataChunk.labels);
                        acc = acc/testdataChunk.dataSize; 
                    elseif mod(obj.descentOpts.iter, 600)== 0
                        preds = model.forward(testdataChunk.inputs);
                        [~, maxInd] = max(preds);
                        acc = 0; 
                        acc = acc+sum(maxInd'==testdataChunk.labels);
                        acc = acc/testdataChunk.dataSize; 
                    end             

                      
                    
                end               
                dataChunk = data();
                
                

                
                
            end
            
            
                 
         
            
            

        end

%         function acc = evaluate(obj,model,data)
%             acc = 0;
%             dataChunk = data();
%             while dataChunk.size > 0
%                 if strcmp(obj.name,'SGLD\_RMSprop');
%                    
%                     preds_acc = model.forward(dataChunk.inputs)* obj.descentOpts.learningRate + obj.ave_preds*obj.descentOpts.learningRate_acc;
%                     obj.descentOpts.learningRate_acc  = obj.descentOpts.learningRate_acc + obj.descentOpts.learningRate;
%                     preds     = preds_acc/obj.descentOpts.learningRate_acc;
%                     obj.ave_preds = preds;
%  
%                 end
% 
%                 if strcmp(obj.name,'RMSprop');
%                     preds = model.forward(dataChunk.inputs);
%                 end                
%                 
%                 [~, maxInd] = max(preds);
%                 acc = acc+sum(maxInd'==dataChunk.labels);
%                 dataChunk = data();
%             end
%             acc = acc/dataChunk.dataSize;
%         end

        %%
        function sobj = saveobj(obj)
            sobj = saveobj@Experiment(obj);
            sobj.data.clear();
            sobj.model.sanitize();
            sobj.descentOpts = obj.descentOpts;
            sobj.optim = obj.optim;
            sobj.optimState = obj.optimState;
            sobj.results = obj.results;
        end
        
        function obj = reload(obj,sobj)
            obj = reload@Experiment(obj,sobj);
            obj.descentOpts = sobj.descentOpts;
            obj.optim = sobj.optim;
            obj.optimState = sobj.optimState;
            obj.results = sobj.results;
        end
        
    end
    methods (Static)
         function obj = loadobj(sobj)
            obj = SimpleExperiment(sobj.name,sobj.modelFactory,sobj.data,sobj.optim);
            obj.reload(sobj);
        end
    end
    
end