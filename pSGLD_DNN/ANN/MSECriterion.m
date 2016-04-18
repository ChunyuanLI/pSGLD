classdef MSECriterion < NNModule
    properties
        average = false;
    end
     methods
         
        function output=forward(self, input, target)
            self.output = 0.5*sum(sum((target-input).^2));
            if self.average
               self.output = self.output/numel(input);
            end
            output=self.output;
        end
        
        function gradInput=backward(self,input,target)
            gradInput = self.updateGradInput(input,target);
        end
        

        function gradInput=updateGradInput(self, input, target)
            self.gradInput = (input-target);
            gradInput = self.gradInput;
        end
     end
end