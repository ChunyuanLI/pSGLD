classdef Sequential < NNModule
    properties
        gpu = false;
        precision = 'double';
        parameters;
        gradient;
    end
    methods
        
        function self=Sequential(opts)
            if nargin>0
                self.readOptions(opts);
            end
        end
        
        function add(self, module)
            self.modules = [self.modules {module}];
            self.parameterSize = self.parameterSize + module.parameterSize;
        end
        
        function applyLayerwise(self,fun)
            arrayfun(fun,self.modules);
%             for i=1:length(self.modules)
%                 fun(self.modules{i});
%             end
        end
        
        function sanitize(self)
            function clearFields(layer)
                layer = layer{1};
                
                if isprop(layer,'output'); layer.output = []; end
                if isprop(layer,'gradInput'); layer.gradInput = []; end
                if isprop(layer,'accWeightGrads'); layer.accWeightGrads = []; end
                if isprop(layer,'accBiasGrads'); layer.accBiasGrads = []; end
                if isprop(layer,'parameters'); layer.accWeightGrads = []; end
                if isprop(layer,'gradient'); layer.accBiasGrads = []; end
                
                if isa(layer,'Sequential'); layer.sanitize(); end
            end
            self.applyLayerwise(@clearFields);
        end
        %
        %         function useGpu(self,gpu)
        %             self.gpu = gpu;
        %             function  fun(m)
        %                 m.gpu = gpu;
        %             end
        %             self.applyLayerwise(fun);
        %         end
        
        function output=forward(self, input)
            output = input;
            for i=1:length(self.modules)
                output = self.modules{i}.forward(output);
            end
            self.output = output;
        end
        
        function output=aveage_predict(self, data)
            output = data.inputs;
            for i=1:length(self.modules)
                output = self.modules{i}.forward(output);
            end
            self.output = output;
            output = double(output) + data.preds;
            data.preds = output;
        end
        
        function gradInput=updateGradInput(self, input, gradInput)
            for i=length(self.modules):-1:2
                gradInput = self.modules{i}.backward(self.modules{i-1}.output, gradInput);
            end
            gradInput = self.modules{1}.backward(input, gradInput);
            self.gradInput = gradInput;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end
        
        
        function updateParameters(self,opts)
            if nargin < 2; opts = {}; end
            for i=length(self.modules):-1:1
                self.modules{i}.updateParameters(opts);
            end
        end
        
        function parameters=getParameters(self)
            if isempty(self.parameters)
                self.parameters = zeros(self.parameterSize,1,self.precision);
                if self.gpu; self.parameters = gpuArray(self.parameters); end
            end
            from = 1;
            for i=1:length(self.modules)
                if self.modules{i}.parameterSize > 0
                    to = from + self.modules{i}.parameterSize - 1;
                    self.parameters(from:to) = self.modules{i}.getParameters();
                    from = to+1;
                end
            end
            parameters = self.parameters;
        end
        
        function setParameters(self,parameters)
            from = 1;
            for i=1:length(self.modules)
                if self.modules{i}.parameterSize > 0
                    to = from + self.modules{i}.parameterSize - 1;
                    self.modules{i}.setParameters(parameters(from:to));
                    from = to+1;
                end
            end
        end
        
        function gradient = getParametersGradient(self)
            if isempty(self.gradient)
                self.gradient = zeros(self.parameterSize,1,self.precision);
                if self.gpu; self.gradient = gpuArray(self.gradient); end
            end
            from = 1;
            for i=1:length(self.modules)
                if self.modules{i}.parameterSize > 0
                    to = from + self.modules{i}.parameterSize - 1;
                    self.gradient(from:to) = self.modules{i}.getParametersGradient();
                    from = to+1;
                end
            end
            gradient = self.gradient;
        end
        
        function setParametersGradient(self,parameters)
            from = 1;
            for i=1:length(self.modules)
                if self.modules{i}.parameterSize > 0
                    to = from + self.modules{i}.parameterSize - 1;
                    self.modules{i}.setParametersGradient(parameters(from:to));
                    from = to+1;
                end
            end
        end
        
    end
    
end