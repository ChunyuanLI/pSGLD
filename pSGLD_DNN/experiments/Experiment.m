classdef Experiment < handle
    properties
        name;
        savePath = '';
        randSeed = 101;
        
        initialized = false;
        forceInit = false;        
        
        modelFactory;        
        model;
        hasModel = false;
        forceLoadModel = false;
        
        hasData = false;
        forceLoadData = false;
        dataFactory;
        data;
        
        environmentVariablesSetter;
    end
    methods (Abstract)
        runExperiment(obj);
    end
    methods
        
        function obj = Experiment(name, modelFactory, data)
            obj.name = name;
            obj.modelFactory = modelFactory;
            obj.data = data;
            
            obj.environmentVariablesSetter = @environmentVariables;
        end

        function init(obj)
            obj.environmentVariablesSetter();
            rng(obj.randSeed,'twister');
            
            if ~obj.hasData || obj.forceLoadData
                obj.data.load();
                obj.hasData = true;
            end
            
            if ~obj.hasModel || obj.forceLoadModel
                obj.model = obj.modelFactory();
                obj.hasModel = true;
            end
        end
        function run(obj)
            
            if ~obj.initialized || obj.forceInit
                obj.init();
                obj.initialized = true;
            end
            obj.runExperiment();
            obj.save();
        end
        
        function finish(obj)
        end
        
        function save(obj)
            saveas = strcat(obj.savePath,obj.name, ...
                '_learningRate_', num2str(obj.descentOpts.learningRate),...
                '_batchSize_', num2str(obj.descentOpts.batchSize),'.mat');
            fprintf('Saving %s...\n',saveas);
            descentOpts = obj.descentOpts;
            results = obj.results;
            model   = obj.model;
           % save(saveas,'-v7.3','descentOpts','results', 'model');
           %  save(saveas,'-v7.3','descentOpts','results', 'model');
        end
        
        function sobj = saveobj(obj)
            sobj.name = obj.name;
            sobj.data = obj.data;
            sobj.modelFactory = obj.modelFactory;
            sobj.model = obj.model;
            sobj.randSeed = obj.randSeed;
        end
        
        function obj = reload(obj,sobj)
            obj.model = sobj.model;
            obj.randSeed = sobj.randSeed;
        end
        
    end
    methods (Static)
         function obj = loadobj(sobj)
            obj = Experiment(sobj.name,sobj.modelFactory,sobj.data);
            obj.reload(sobj);
        end
    end
    
end