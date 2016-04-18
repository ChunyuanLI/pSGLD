classdef LogSoftmax < NNModule
    methods

        function output=forward(self,x)
            c=max(x);
            lse=log(sum(exp(bsxfun(@minus,x,c))))+c;
            self.output = bsxfun(@minus,x,lse);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            self.gradInput = gradOutput - bsxfun(@times,exp(self.output),sum(gradOutput));
            gradInput = self.gradInput;
        end
    end
end