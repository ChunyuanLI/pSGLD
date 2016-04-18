classdef SaddleExperiment < SimpleExperiment
    properties
        criterion=NLLCriterion();
    end
    methods
        function obj = SaddleExperiment(name, modelFactory, dataFactory, optim)
            obj = obj@SimpleExperiment(name, modelFactory, dataFactory, optim);
        end
        function err=getValue(obj)
            data=obj.data.train;
            preds = obj.model.forward(data.inputs);
            err = obj.criterion.forward(preds,data.labels);
        end
        function G=getGrad(obj,layer)
            oP=obj.model.getParameters;
            data=obj.data.train;
            preds = obj.model.forward(data.inputs);
            grads = obj.criterion.backward(preds,data.labels);
            grads = grads./size(preds,2); % average according to batch size
            obj.model.backward(data.inputs,grads);
            if nargin<2;
                layer=1;
            end
            %             if nargin<2
            %             obj.optimState = obj.optim(obj.model,obj.descentOpts,obj.optimState);
            %             P=obj.model.getParameters;
            %             G=(P-oP)./obj.descentOpts.learningRate;
            %             obj.model.setParameters(oP);
            %             else
            G=obj.model.modules{1+(layer-1)*2}.getParametersGradient;
            %             end
        end
        function vals=evalLine(obj,direction,steps,layer)
            if nargin<4
                layer=1;
            end
            ii=1+(layer-1)*2;
            oP=obj.model.modules{ii}.getParameters;
            S=numel(steps);
            vals=numel(steps);
            for s=1:S
                obj.model.modules{ii}.setParameters(oP+steps(s)*direction);
                vals(s)=obj.getValue;
            end
            obj.model.modules{ii}.setParameters(oP);
        end
        function vals=evalGrid(obj,d1,d2,steps1,steps2,layer)
            if nargin<6
                layer=1;
            end
            ii=1+(layer-1)*2;
            oP=obj.model.modules{ii}.getParameters;
            S1=numel(steps1);
            S2=numel(steps2);
            vals=zeros(S1,S2);
            for s1=1:S1
                for s2=1:S2
                obj.model.modules{ii}.setParameters(oP+steps1(s1)*d1+steps2(s2)*d2);
                vals(s1,s2)=obj.getValue;
                end
            end
            obj.model.modules{ii}.setParameters(oP);
        end
    end
    
end