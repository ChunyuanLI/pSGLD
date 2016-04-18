classdef Sigmoid < NNModule
    methods
        function output=forward(self,x)
            self.output = logsig(x);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = (1-self.output).*self.output.*gradOutput;
            gradInput = self.gradInput;
        end
    end
end