classdef Dataset < handle
    properties
        unusedFiles;
        usedFiles;
        parent;
        inputs;
        labels;
        shuffle = false;
        name;
        preds;
    end
    methods
        
        function obj = Dataset(parent,name,files)
            obj.parent = parent;
            obj.name = name;
            obj.unusedFiles = files;
        end
        
        function loadDataToRAM(obj, N)
            M = N;
            
            while M>0 && numel(obj.unusedFiles)>0
                [file,ii] = datasample(obj.unusedFiles,1);
                batch = obj.parent.loadDataFile(file{1});
                obj.inputs = [obj.inputs ; batch.inputs];
                obj.labels = [obj.labels ; batch.labels];
                M = M-numel(batch.labels);
                obj.usedFiles{end+1} = file;
                obj.unusedFiles(ii) = [];
            end
            obj.preds = sparse(length(unique(obj.labels)),N);
            obj.parent.preprocessData(obj);
            if obj.shuffle
                sh = randperm(numel(obj.labels));
                obj.inputs = obj.inputs(:,sh(1:N));
                obj.labels = obj.labels(sh(1:N));
            else
                obj.inputs = obj.inputs(:,1:N);
                obj.labels = obj.labels(1:N);
            end
        end
        
        function iter = getIterator(obj,chunkSize)
            if nargin<2 || chunkSize<1; chunkSize = numel(obj.labels); end
            lo = 1;
            iter = @iterator;
            ind = 1;
            function chunk = iterator()
                if lo > numel(obj.labels)
                    chunk = obj.getChunk(0,0);
                else
                    if lo + chunkSize - 1 < numel(obj.labels)
                        if lo + 2*chunkSize - 1 < numel(obj.labels)
                            hi = lo + chunkSize - 1;
                        else
                            hi = lo + floor(.5*(numel(obj.labels)-lo));
                        end
                    else
                        hi = numel(obj.labels);
                    end
                        
                    chunk = obj.getChunk(lo,hi);
                    chunk.ind = ind;
                    ind=ind+1;
                    lo = hi+1;
                end
            end
            
        end
        
        function chunk = getChunk(obj,lo,hi)
            if nargin == 1
                lo = 1;
                hi = numel(obj.labels);
            elseif nargin == 2
                hi = lo;
                lo = 1;
            end
            
            if lo < 1 || lo > numel(obj.labels)
                chunk.empty = true;
                chunk.size = 0;
            else
                chunk.empty = false;
                chunk.lo = lo;
                chunk.hi = hi;
                chunk.inputs = obj.inputs(:,lo:hi);
                chunk.labels = obj.labels(lo:hi);
                chunk.size = numel(chunk.labels);
                chunk.preds = obj.preds(:,lo:hi);
            end
            chunk.dataSize = numel(obj.labels);
            chunk = obj.parent.processChunk(chunk);
        end
        
        function updatePreds(obj, preds)
            obj.preds(:,lo:hi) = preds;
        end
    end
end