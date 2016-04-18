classdef Reshape < NNModule
    properties
        from
        to
    end
    methods
        function self=Reshape(from, to)
            self.from = from;
            self.to = to;
        end
        
        function output=forward(self,x)
            self.output = reshape(x,self.to);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = reshape(gradOutput,self.from);
            gradInput = self.gradInput;
        end
    end
end