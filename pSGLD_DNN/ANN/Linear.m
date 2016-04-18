classdef Linear < NNModule
    properties
        weights
        bias
        accWeightGrads
        accBiasGrads
        inSize
        outSize
        weightsStepSize = 1;
        biasStepSize = 1;
        precision = 'double';
        gpu = false;
    end
    
     methods
         
        
        function self=Linear(inSize, outSize, opts)
            
            if nargin > 2
                self.readOptions(opts);
            end
            
            self.inSize = inSize;
            self.outSize = outSize;
            
            self.output = zeros(self.outSize,1);
            self.gradInput = zeros(self.inSize,1);
            
            self.accWeightGrads = zeros(self.outSize, self.inSize, self.precision);
            self.accBiasGrads = zeros(self.outSize, 1, self.precision);
            
            r  = sqrt(6) / sqrt(self.inSize+self.outSize+1);
            self.weights = rand(self.outSize, self.inSize, self.precision) * 2 * r - r;
            self.bias = rand(self.outSize, 1, self.precision) * 2 * r - r;
            
            self.parameterSize = self.outSize*(self.inSize+1);
            
            if self.gpu
                self.output = gpuArray(self.output);
                self.gradInput = gpuArray(self.gradInput);
                self.accWeightGrads = gpuArray(self.accWeightGrads);
                self.accBiasGrads = gpuArray(self.accBiasGrads);
                self.weights = gpuArray(self.weights);
                self.bias = gpuArray(self.bias);
            end
            
        end

        function output=forward(self, input)
            self.output = bsxfun(@plus,self.weights*input,self.bias);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
            self.accGradParameters(input, gradOutput);
        end
        
        function gradInput=updateGradInput(self, input, gradOutput)
            self.gradInput = self.weights'*gradOutput;
            gradInput = self.gradInput;
        end
       
        
        function accGradParameters(self, input, gradOutput)
            self.accWeightGrads = self.accWeightGrads + gradOutput*input';
            self.accBiasGrads = self.accBiasGrads + sum(gradOutput,2);
        end
        
        function updateParameters(self,opts)
            if isfield(opts,'zeroWeights')
                zeroWeights = opts.zeroWeights;
            else
                zeroWeights = true;
            end
            
            if isfield(opts,'scale')
                scale = opts.scale;
            else
                scale = 1;
            end
            
            if isfield(opts,'weightsStepSize')
                weightsStep = opts.weightsStepSize;
            else
                weightsStep = self.weightsStepSize;
            end
            
            if isfield(opts,'biasStepSize')
                biasStep = opts.biasStepSize;
            else
                biasStep = self.biasStepSize;
            end
            
            
            self.weights = self.weights - weightsStep*self.accWeightGrads;
            self.bias = self.bias - biasStep*self.accBiasGrads;
            if zeroWeights
                self.zeroParameters();
            end
        end
        
        function zeroParameters(self)
            self.accWeightGrads = 0*self.accWeightGrads;
            self.accBiasGrads =  0*self.accBiasGrads;
        end
        
        function parameters=getParameters(self)
            parameters = [self.weights(:) ; self.bias(:)];
        end
        
        function parameters=setParameters(self, parameters)
            self.weights = reshape(parameters(1:(self.inSize*self.outSize)),self.outSize,self.inSize);
            self.bias = reshape(parameters((self.inSize*self.outSize)+1:end),self.outSize, 1);
        end
        
        function gradient = getParametersGradient(self)
            gradient = [self.accWeightGrads(:) ; self.accBiasGrads(:)];
        end
        
        function parameters=setParametersGradient(self, parameters)
            self.accWeightGrads = reshape(parameters(1:(self.inSize*self.outSize)),self.outSize,self.inSize);
            self.accBiasGrads = reshape(parameters((self.inSize*self.outSize)+1:end),self.outSize, 1);
        end
        
     end    
end