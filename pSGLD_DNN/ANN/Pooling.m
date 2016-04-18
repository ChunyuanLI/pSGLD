classdef Pooling < NNModule
    properties
        pWidth
        pHeight
        mask
        %ssWidth
        %ssHeight
        padding = [0 0 0 0]; %top bottom left right
        stride = [1 1]; % vertical horizontal
        method = 'max'; %or 'avg'
    end
    methods
        function self=Pooling(pHeight, pWidth)
            self.pWidth = pWidth;
            self.pHeight = pHeight;
            %self.ssWidth = ssWidth;
            %self.ssHeight = ssHeight;
        end
        
        function sizes = getOutputSize(self,sizes)
            inWidth = sizes(1);
            inHeight = sizes(2);
            outHeight = floor((inHeight + self.padding(1) + self.padding(2) - self.pHeight)/self.stride(1)) + 1;
            outWidth = floor((inWidth + self.padding(3) + self.padding(4) - self.pWidth)/self.stride(2)) + 1;            
            sizes = [outWidth, outHeight];
        end
        
        function noOverlap(self)
            self.stride = [self.pHeight self.pWidth];
        end
        
        function output=forward(self,input)
             self.output = vl_nnpool(input,[self.pHeight, self.pWidth],'Pad',self.padding,'Stride',self.stride);
            output=self.output;
        end
        
        function gradInput=backward(self,input,gradOutput)
            gradInput = self.updateGradInput(input,gradOutput);
        end

        function gradInput=updateGradInput(self,input,gradOutput)
            gradInput = vl_nnpool(input,[self.pHeight, self.pWidth],gradOutput,'Pad',self.padding,'Stride',self.stride);
            self.gradInput = gradInput;
        end
    end
end