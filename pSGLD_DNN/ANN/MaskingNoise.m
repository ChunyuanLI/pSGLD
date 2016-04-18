classdef MaskingNoise < NNModule
    properties
        level
        maskInd
    end
    
     methods
        
        function self=MaskingNoise(level)
            self.level = level;
        end

        function output=forward(self, input)
            s = floor(numel(input)*self.level);
            self.maskInd = randsample(numel(input),s);
            self.output = input;
            self.output(self.maskInd) = zeros(length(self.maskInd),1);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end
        
          function gradInput=updateGradInput(self, input, gradOutput)
            self.gradInput = gradOutput;
            self.gradInput(self.maskInd) = zeros(length(self.maskInd),1);
            gradInput = self.gradInput;
          end  
     end    
end