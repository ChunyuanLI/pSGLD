classdef Mul < handle
    properties
        output
        gradInput
        weights
        accWeightGrads
        size
        constant
    end
    
     methods
        
        function self=Mul(lsize)
            self.size = lsize;
            self.output = zeros(self.size,1);
            self.gradInput = zeros(self.size,1);
            self.accWeightGrads = zeros(self.size, 1);
            self.weights = ones(self.size, 1);
            self.constant = 1;
        end

        function output=forward(self, input)
            self.output = self.constant*self.weights.*input;
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
            self.accGradParameters(input, gradOutput);
        end
        
          function gradInput=updateGradInput(self, input, gradOutput)
            self.gradInput = self.constant*self.weights.*gradOutput;
            gradInput = self.gradInput;
          end
       
        
        function accGradParameters(self, input, gradOutput)
            self.accWeightGrads = self.accWeightGrads + self.constant*gradOutput.*input;
        end
        
        function updateParameters(self,learningRate,opts)
            zeroWeights = true;
            batchSize = 1;
            weightDecay = 0;
            
            if isfield(opts,'zeroWeights')
                zeroWeights = opts.zeroWeights;
            end
            if isfield(opts,'batchSize')
                batchSize = opts.batchSize;
            end
            if isfield(opts,'weightDecay')
                weightDecay = opts.weightDecay;
            end
            
            if weightDecay > 0
                self.accWeightGrads = self.accWeightGrads+weightDecay*self.weights;
            end
            self.weights = self.weights - learningRate*self.accWeightGrads/batchSize;
            if zeroWeights
                self.accWeightGrads = zeros(self.size, 1);
            end
        end
        
        function parameters=getParameters(self)
            parameters = self.weights;
        end
        
        function parameters=setParameters(self, parameters)
            self.weights = parameters;
        end
        
        function gradient = getParametersGradient(self)
            gradient = self.accWeightGrads;
        end
        
        function s=parameterSize(self)
            s=self.size;
        end
        
     end    
end