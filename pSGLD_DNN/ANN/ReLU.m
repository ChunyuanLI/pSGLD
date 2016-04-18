classdef ReLU < NNModule
    methods
        function output=forward(self,x)
            self.output = max(x,0);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = (self.output>0).*gradOutput;
            gradInput = self.gradInput;
        end
    end
end