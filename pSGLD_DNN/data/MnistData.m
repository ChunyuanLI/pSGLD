classdef MnistData < handle
    properties
        trainSizeMax = 60000;
        testSizeMax = 10000;
        trainSizeCurrent;
        testSizeCurrent;
        inSize = [28 28 1]; %width height #channels
        outSize = 10;
        trainDataInputsPath = 'data/mnistData/train-images.idx3-ubyte';
        trainDataLabelsPath = 'data/mnistData/train-labels.idx1-ubyte';
        testDataInputsPath = 'data/mnistData/t10k-images.idx3-ubyte';
        testDataLabelsPath = 'data/mnistData/t10k-labels.idx1-ubyte';
        precision = @double;
        flatten = true;
        gpu = false;
        whiten = true;
        train;
        test;
        hasData = false;
        mean;
        whM;
        whMInv;
    end
    methods
        function obj = MnistData(Ntrain,Ntest,opts)
            
            
            if nargin<1; Ntrain  = obj.trainSizeMax; end
            if nargin<2; Ntest   = obj.testSizeMax; end
            if nargin > 2; obj.readOptions(opts); end
            
            obj.trainSizeCurrent = min(Ntrain,obj.trainSizeMax);
            obj.testSizeCurrent = min(Ntest,obj.testSizeMax);
            
            % Training set files
            files = cell(1);
            files{1}.inputs = obj.trainDataInputsPath;
            files{1}.labels = obj.trainDataLabelsPath;
            obj.train = Dataset(obj,'train',files);
            
            % Test set files
            files = cell(1);
            files{1}.inputs = obj.testDataInputsPath;
            files{1}.labels = obj.testDataLabelsPath;
            obj.test = Dataset(obj,'test',files);
            
        end
        
        function load(obj)          
            if ~obj.hasData

                obj.train.loadDataToRAM(obj.trainSizeCurrent);
                obj.test.loadDataToRAM(obj.testSizeCurrent);
                
                obj.hasData = true;
            end
        end
        
        
        function iter = getIterator(obj,data)
            iter = @iterator;
            function chunk = iterator(n)
                n = min(n,numel(data));
                chunk = data{n};
                if obj.gpu; chunk.inputs = gpuArray(chunk.inputs); end
                
                if ~obj.flatten
                    chunk.inputs = reshape(chunk.inputs, [obj.inSize size(chunk,2)]);
                end
            end
        end
        
        
        function clear(obj)
            obj.train = [];
            obj.test = [];
            obj.hasData = false;
        end
        
        
        function batch = loadDataFile(obj,paths)
            batch.inputs = loadMNISTImages(paths.inputs);
            batch.labels = loadMNISTLabels(paths.labels)+1;
        end
        
        function preprocessData(obj,dataset)
            dataset.inputs = obj.precision(dataset.inputs);          
        end

        function chunk = processChunk(obj,chunk)
            if chunk.size > 0
                if obj.gpu
                    chunk.inputs = gpuArray(chunk.inputs);
                    chunk.labels = gpuArray(chunk.labels);
                end

                if ~obj.flatten
                    chunk.inputs = reshape(chunk.inputs,[obj.inSize numel(chunk.labels)]);
                end
            end
            chunk.inSize = obj.inSize;
        end
        
        function readOptions(self,opts)
            fields = fieldnames(opts);
            for ii=1:numel(fields)
                f = fields{ii};
                if isprop(self,f)
                    self.(f) = opts.(f);
                end
            end
        end  
    end
end

% function rdata =  MnistData(N,opts)
%     maxN = 60000;
%     trainDataInputsPath = 'data/mnistData/train-images.idx3-ubyte';
%     trainDataLabelsPath = 'data/mnistData/train-labels.idx1-ubyte';
%     testDataInputsPath = 'data/mnistData/t10k-images.idx3-ubyte';
%     testDataLabelsPath = 'data/mnistData/t10k-labels.idx1-ubyte';
%     
%     if nargin < 2;opts = {};end
%     if ~isfield(opts,'precision'); opts.precision = @double; end
%     if ~isfield(opts,'flatten'); opts.flatten = true; end
%     if ~isfield(opts,'gpu'); opts.gpu = false; end
%     
%     if nargin<1
%         N = maxN;
%     end
%     N = min(N,maxN);
%     
% 
%     persistent data;
% 
%     if isempty(data) || numel(data.train.labels)~=N
%         data.inSize = [28 28 1]; %width height #channels
%         data.outSize = 10;
%         
%         %% Load training set
%         data.train.inputs = loadMNISTImages(trainDataInputsPath);
%         data.train.labels = loadMNISTLabels(trainDataLabelsPath)+1;
%         
%         data.train.inputs = opts.precision(data.train.inputs(:,1:N));
%         data.train.labels = data.train.labels(1:N);
%         
%         %% Load test set
%         data.test.inputs = opts.precision(loadMNISTImages(testDataInputsPath));
%         data.test.labels = loadMNISTLabels(testDataLabelsPath)+1;
%    
%     
%         %% Preprocessing
%         
%         if opts.gpu
%             data.train.inputs = gpuArray(data.train.inputs);
%             data.train.labels = gpuArray(data.train.labels);
%             data.test.inputs = gpuArray(data.test.inputs);
%             data.test.labels = gpuArray(data.test.labels);
%         end
%         
%         if ~opts.flatten
%             data.train.inputs = reshape(data.train.inputs,[data.inSize N]);
%             data.test.inputs = reshape(data.test.inputs,[data.inSize size(data.test.inputs,2)]);
%         end
%         
%     end
% 
%     rdata = data;
% end
