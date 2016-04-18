classdef NNModule < handle
    
    properties
        output=[];
        gradInput=[];
        modules={};
        parameterSize=0;
    end
    methods (Abstract)
        forward(self,x);
        gradInput=backward(self,input,gradOutput);
        gradInput=updateGradInput(self,input,gradOutput);
    end
    
    methods
        
        function updateParameters(self,learningRate,batchSize,zeroWeights)
        end
        
        function parameters=getParameters(self)
            parameters = [];
        end
        
        function setParameters(self, parameters)
        end
        
        function parameters=getParametersGradient(self)
            parameters = [];
        end
        
        function setParametersGradient(self, parameters)
        end
        
        function readOptions(self,opts)
            fields = fieldnames(opts);
            for ii=1:numel(fields)
                f = fields{ii};
                if isprop(self,f)
                    self.(f) = opts.(f);
                end
            end
        end
        
    end
end