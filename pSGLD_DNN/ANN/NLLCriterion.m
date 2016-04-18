classdef NLLCriterion < NNModule
     methods
        function output=forward(self, input, target)
            ind = sub2ind(size(input),target,(1:size(input,2))');
            self.output = sum(-input(ind));
            output=self.output;
        end

        function gradInput=backward(self,input,target)
            gradInput = self.updateGradInput(input,target);
        end
        
        function gradInput=updateGradInput(self, input, target)
            self.gradInput = zeros(size(input));
            ind = sub2ind(size(input),target,(1:size(input,2))');
            self.gradInput(ind) = -1;
            gradInput = self.gradInput;
        end
     end
end