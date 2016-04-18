classdef Flatten < NNModule
    properties
        flatLastDim = false;
    end
    methods
        
        function self=Flatten(flatLastDim)
            if nargin>0
                self.flatLastDim = flatLastDim;
            end
        end
        
        function output=forward(self,input)
            if self.flatLastDim
                self.output = input(:);
            else
            sizes = size(input);
            self.output = reshape(input,[],sizes(end)); %flatten all dimensions except the last (usually the batch dimension)
            end
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = reshape(gradOutput,size(input));
            gradInput = self.gradInput;
        end
    end
end