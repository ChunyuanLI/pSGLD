classdef Tanh < NNModule
    methods
        function output=forward(self,x)
            self.output = tanh(x);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = (1-self.output.^2).*gradOutput;
            gradInput = self.gradInput;
        end
    end
end