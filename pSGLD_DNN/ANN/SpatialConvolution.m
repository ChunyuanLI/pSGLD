classdef SpatialConvolution < NNModule
    properties
        weights
        bias
        accWeightGrads
        accBiasGrads
        inChannels
        %inWidth
        %inHeight
        outChannels
        filterWidth
        filterHeight
        outSize
        weightsStepSize = 1;
        biasStepSize = 1;
        precision = 'double';
        gpu = false;
        padding = [0 0 0 0]; %top bottom left right
        stride = [1 1]; % vertical horizontal
    end
    
     methods
         
        
        function self=SpatialConvolution(inChannels, outChannels, filterHeight, filterWidth, opts)
           if nargin > 4
                self.readOptions(opts);
            end
            
            
            addpath([getenv('MATCONVNET_PATH') '/matlab']);
            vl_setupnn();
            self.inChannels = inChannels;
            %self.inWidth = inWidth;
            %self.inHeight = inHeight;
            self.outChannels = outChannels;
            self.filterWidth = filterWidth;
            self.filterHeight = filterHeight;
            
            self.output;
            self.gradInput;
            
            self.accWeightGrads = zeros(self.filterHeight,self.filterWidth,self.inChannels,self.outChannels, self.precision);
            self.accBiasGrads = zeros(self.outChannels, 1, self.precision);
            
            r  = sqrt(6) / sqrt((self.inChannels+self.outChannels)*self.filterWidth*self.filterHeight);
            self.weights = rand(self.filterHeight,self.filterWidth,self.inChannels,self.outChannels,self.precision) * 2 * r - r;
            self.bias = rand(self.outChannels, 1, self.precision) * 2 * r - r;
            
            self.parameterSize = numel(self.weights) + numel(self.bias);
            
            if self.gpu
                %self.output = gpuArray(self.output);
                %self.gradInput = gpuArray(self.gradInput);
                self.accWeightGrads = gpuArray(self.accWeightGrads);
                self.accBiasGrads = gpuArray(self.accBiasGrads);
                self.weights = gpuArray(self.weights);
                self.bias = gpuArray(self.bias);
            end
            
        end

        function output=forward(self, input)
            self.output = vl_nnconv(input,self.weights,self.bias,'Pad',self.padding,'Stride',self.stride);
            output=self.output;
        end
        
        function sizes = getOutputSize(self,sizes)
            inHeight = sizes(1);
            inWidth = sizes(2);
            outHeight = floor((inHeight + self.padding(1) + self.padding(2) - self.filterHeight)/self.stride(1)) + 1;
            outWidth = floor((inWidth + self.padding(3) + self.padding(4) - self.filterWidth)/self.stride(2)) + 1;            
            sizes = [outHeight, outWidth];
        end
        
        function tightPadding(self)
            self.padding = [self.filterHeight-1 self.filterHeight-1 self.filterWidth-1 self.filterWidth-1];
        end
        
        function gradInput=backward(self,input,gradOutput)
            [self.gradInput, weightGrads, biasGrads] = vl_nnconv(input,self.weights,self.bias,gradOutput,'Pad',self.padding,'Stride',self.stride);
            self.accWeightGrads = self.accWeightGrads + weightGrads;
            self.accBiasGrads = self.accBiasGrads + biasGrads;
            gradInput = self.gradInput;
            %gradInput = self.updateGradInput(input,gradOutput);
            %self.accGradParameters(input, gradOutput);
        end
        
        function gradInput=updateGradInput(self, input, gradOutput)
            disp('Call backward() for gradInput of SpatialConvolution.');
            gradInput = self.gradInput;
        end
       
        
        function accGradParameters(self, input, gradOutput)
            disp('Call backward() for gradient accumulation for SpatialConvolution.');
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
            
            
            self.weights = self.weights - weightsStep*self.accWeightGrads*scale;
            self.bias = self.bias - biasStep*self.accBiasGrads*scale;
            if zeroWeights
                self.accWeightGrads = 0*self.accWeightGrads;
                self.accBiasGrads =  0*self.accBiasGrads;
            end
        end
        
        function parameters=getParameters(self)
            parameters = [self.weights(:) ; self.bias(:)];
        end
        
        function parameters=setParameters(self, parameters)
            self.weights = reshape(parameters(1:numel(self.weights)),size(self.weights));
            self.bias = reshape(parameters(numel(self.weights)+1:end),size(self.bias));
        end
        
        function gradient = getParametersGradient(self)
            gradient = [self.accWeightGrads(:) ; self.accBiasGrads(:)];
        end
        
        function parameters=setParametersGradient(self, parameters)
            self.accWeightGrads = reshape(parameters(1:numel(self.accWeightGrads)),size(self.accWeightGrads));
            self.accBiasGrads = reshape(parameters(numel(self.accWeightGrads)+1:end),size(self.accBiasGrads));
        end
        
     end    
end