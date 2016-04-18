classdef Cifar10Data < handle
    properties
        numBatches = 5;
        trainSizeMax = 50000;
        testSizeMax = 10000;
        trainSizeCurrent;
        testSizeCurrent;
        inSize = [32 32 3]; %width height #channels
        outSize = 10;
        trainDataPath = 'data/cifar10Data/data_batch_%d.mat';
        testDataPath = 'data/cifar10Data/test_batch.mat';
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
        function obj = Cifar10Data(Ntrain,Ntest,opts)
            
            
            if nargin<1; Ntrain  = obj.trainSizeMax; end
            if nargin<2; Ntest   = obj.testSizeMax; end
            if nargin > 2; obj.readOptions(opts); end
            
            obj.trainSizeCurrent = min(Ntrain,obj.trainSizeMax);
            obj.testSizeCurrent = min(Ntest,obj.testSizeMax);
            
            % Training set files
            files = cell(obj.numBatches,1);
            for ii=1:obj.numBatches
                files{ii} = sprintf(obj.trainDataPath,ii);
            end
            obj.train = Dataset(obj,'train',files);
            
            % Test set files
            files = {obj.testDataPath};
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
        
        
        function batch = loadDataFile(obj,path)
            data = load(path);
            batch.inputs = data.data;
            batch.labels = data.labels+1;
        end
        
        function preprocessData(obj,dataset)
            dataset.inputs = obj.precision(dataset.inputs')/255;
            if obj.whiten
                warning('off','MATLAB:warn_r14_stucture_assignment');
                if strcmp(dataset.name,'train')
                    [dataset.inputs,obj.mean,obj.whMInv,obj.whM] = whiten(dataset.inputs);
                else %test
                    dataset.inputs = obj.whM*bsxfun(@minus, dataset.inputs, obj.mean);
                end
            end            
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
% function rdata =  Cifar10Data(N,opts)
% numBatches = 5;
% samplesPerBatch = 10000;
% trainDataPath = 'data/cifar10Data/data_batch_%d.mat';
% testDataPath = 'data/cifar10Data/test_batch.mat';
%
% if nargin<1
%     N = numBatches*samplesPerBatch;
% end
% N = min(N,numBatches*samplesPerBatch);
%
% if nargin < 2;opts = {};end
%
%
% persistent data;
%
% if isempty(data) || numel(data.train.labels)~=N
%     data.inSize = [32 32 3]; %width height #channels
%     data.outSize = 10;
%
%     %% Load training set
%     nBatch = 1;
%     data.train.inputs = [];
%     data.train.labels = [];
%     M = N;
%     while M > 0
%         batch = load(sprintf(trainDataPath,nBatch));
%         data.train.inputs = [data.train.inputs ; batch.data];
%         data.train.labels = [data.train.labels ; batch.labels+1];
%         M = M-samplesPerBatch;
%         nBatch = nBatch+1;
%     end
%     data.train.inputs = data.train.inputs(1:N,:);
%     data.train.labels = data.train.labels(1:N);
%
%     data.train.inputs = opts.precision(data.train.inputs')/255; %Convert, transpose and normalize
%
%     %% Load test set
%     testBatch = load(testDataPath);
%     data.test.inputs = testBatch.data;
%     data.test.labels = testBatch.labels+1;
%
%     data.test.inputs = opts.precision(data.test.inputs')/255; %Convert, transpose and normalize
%
%     %% Preprocessing
%
%     if opts.whiten
%         warning('off','MATLAB:warn_r14_stucture_assignment');
%         %M = min(N,10000);
%         [data.train.inputs,data.mean,data.whMInv,data.whM] = whiten(data.train.inputs);
%         %data.train.inputs = data.whM*bsxfun(@minus, data.train.inputs, data.mean);
%         data.test.inputs = data.whM*bsxfun(@minus, data.test.inputs, data.mean);
%     end
%
%     if opts.gpu
%         data.train.inputs = gpuArray(data.train.inputs);
%         data.train.labels = gpuArray(data.train.labels);
%         data.test.inputs = gpuArray(data.test.inputs);
%         data.test.labels = gpuArray(data.test.labels);
%     end
%
%     if ~opts.flatten
%         data.train.inputs = reshape(data.train.inputs,[data.inSize N]);
%         data.test.inputs = reshape(data.test.inputs,[data.inSize size(data.test.inputs,2)]);
%     end
% end
% rdata = data;
% end
