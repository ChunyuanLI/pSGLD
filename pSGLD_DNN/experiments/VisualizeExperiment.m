classdef VisualizeExperiment < SimpleExperiment
    properties
        visualize;
    end
    methods
        function obj = VisualizeExperiment(name, modelFactory, dataFactory, optim)
            obj = obj@SimpleExperiment(name, modelFactory, dataFactory, optim);
        end
        
        function finish(obj)
            figure('name',sprintf('%s filters',obj.name));
            [h,~] = obj.visualize(obj.model.modules{1}.weights');
            %save h?
        end
        
    end
    methods (Static)
         function obj = loadobj(sobj)
            obj = VisualizeExperiment(sobj.name,sobj.modelFactory,sobj.dataFactory,sobj.optim);
            obj = obj.reload(sobj);
        end
    end
    
end