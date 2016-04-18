classdef SubsampledLinear < handle
    properties
        output
        gradInput
        weights
        bias
        accWeightGrads
        accBiasGrads
        inSize
        outSize
        outIndices
        inIndices
    end
    
     methods
        
        function self=SubsampledLinear(inSize, outSize)
            self.inSize = inSize;
            self.outSize = outSize;
            
            self.inIndices = 1:self.inSize;
            self.outIndices = 1:self.outSize;
            
            self.output = zeros(self.outSize,1);
            self.gradInput = zeros(self.inSize,1);
            
            self.accWeightGrads = zeros(self.outSize, self.inSize);
            self.accBiasGrads = zeros(self.outSize, 1);
            
            r  = sqrt(6) / sqrt(self.inSize+self.outSize+1);
            self.weights = rand(self.outSize, self.inSize) * 2 * r - r;
            self.bias = rand(self.outSize, 1) * 2 * r - r;
        end

        function output=forward(self, input)
            self.output = self.weights(self.outIndices,self.inIndices)*input + self.bias(self.outIndices);
            output=self.output;
        end
        
        function subsample(self,inIndices,outIndices)
            self.outIndices = outIndices;
            self.inIndices = inIndices;
            if numel(self.outIndices) == 0
                self.outIndices = 1:self.outSize;
            end
            if numel(self.inIndices) == 0
                self.inIndices = 1:self.inSize;
            end
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
            self.accGradParameters(input, gradOutput);
        end
        
        function gradInput=updateGradInput(self, input, gradOutput)
            self.gradInput = self.weights(self.outIndices,self.inIndices)'*gradOutput;
            gradInput = self.gradInput;
        end
       
        
        function accGradParameters(self, input, gradOutput)
            self.accWeightGrads(self.outIndices,self.inIndices) = self.accWeightGrads(self.outIndices,self.inIndices) + gradOutput*input';
            self.accBiasGrads(self.outIndices) = self.accBiasGrads(self.outIndices) + gradOutput;
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
            
            self.weights(self.outIndices,self.inIndices) = self.weights(self.outIndices,self.inIndices) - learningRate*self.accWeightGrads(self.outIndices,self.inIndices)/batchSize;
            self.bias(self.outIndices) = self.bias(self.outIndices) - learningRate*self.accBiasGrads(self.outIndices)/batchSize;
            
            if weightDecay > 0
                self.weights(self.outIndices,self.inIndices) = self.weights(self.outIndices,self.inIndices) - learningRate*weightDecay*self.weights(self.outIndices,self.inIndices);
            end
            
            if zeroWeights
                self.accWeightGrads = zeros(self.outSize, self.inSize);
                self.accBiasGrads = zeros(self.outSize, 1);
            end
        end
        
        function parameters=getParameters(self)
            w = self.weights(self.outIndices,self.inIndices);
            b = self.bias(self.outIndices);
            parameters = [w(:) ; b(:)];
        end
        
        function parameters=setParameters(self, parameters)
            lin = length(self.inIndices);
            lout = length(self.outIndices);
            self.weights(self.outIndices,self.inIndices) = reshape(parameters(1:(lout*lin)),lout,lin);
            self.bias(self.outIndices) = reshape(parameters((lin*lout)+1:end),lout, 1);
        end
        
        function gradient = getParametersGradient(self)
            w = self.accWeightGrads(self.outIndices,self.inIndices);
            b = self.accBiasGrads(self.outIndices);
            gradient = [w(:) ; b(:)];
        end
        
        function s=parameterSize(self)
            s=length(self.outIndices)*(length(self.inIndices)+1);
        end
        
     end    
end