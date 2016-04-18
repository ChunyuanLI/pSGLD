classdef Curves < handle
    methods (Static)

        function curves=aggregate(structure,accessor)
            if numel(structure) > 0
                curves = cell(numel(structure),1);
                if isa(accessor,'function_handle')
                    for ii=1:numel(structure)
                        curves{ii} = accessor(structure{ii});
                    end
                elseif isa(accessor,'char')
                    for ii=1:numel(structure)
                        curves{ii} = structure{ii}.(accessor);
                    end
                end
            end
        end
        
        function flatCurves = flatten(curves)
            if ~isa(curves{1},'cell')
                flatCurves = curves;
                return;
            end
            flatCurves = {};
            for ii=1:numel(curves)
                flatChildren = Curves.flatten(curves{ii});
                for jj=1:numel(flatChildren); flatCurves{end+1} = flatChildren{jj}; end
            end
        end
            
                
        
        function curves=process(curves,processor,attr)
            if nargin < 3
                for ii=1:numel(curves)
                    curves{ii} = processor(curves{ii});
                end
            else
                for ii=1:numel(curves)
                    curves{ii}.(attr) = processor(curves{ii}.(attr));
                end
            end
        end
        
        function result=query(curves,q)
            result = zeros(numel(curves));
            for ii=1:numel(curves)
                result(ii) = q(curves{ii});
            end
        end
        
        function curves = filter(curves,cond)
            curves = curves(cellfun(cond,curves));
            %filteredCurves = {};
            %for ii=1:numel(curves)
            %    if cond(curves{ii}); filteredCurves{end+1} = curves{ii}; end
            %end
        end
        
        function curves = select(curves,attr,values)
            % filter
            cond = @(curve) any(cellfun(@(value) strcmp(curve.(attr),value),values));
            curves = curves(cellfun(cond,curves));
            
            % order
            for ii=1:numel(values)
                matches = find(cellfun(@(curve) strcmp(curve.(attr),values{ii}),curves));
                for jj=1:numel(matches);curves{matches(jj)}.Curves_temp_order = ii;end
            end
            curves = Curves.sort(curves,@(curve) curve.Curves_temp_order);
            curves = Curves.process(curves,@(curve) rmfield(curve,'Curves_temp_order'));
        end

        function curves=findReplace(curves,attr,oldNew)
            if isa(oldNew,'containers.Map')
                renameSet = oldNew;
            elseif isa(oldNew,'cell')
                renameSet = containers.Map('KeyType',class(oldNew{1}{1}),'ValueType',class(oldNew{1}{2}));
                for ii=1:numel(oldNew)
                    renameSet(oldNew{ii}{1}) = oldNew{ii}{2};
                end
            else
                disp('findReplace: don''t know how to interpet input.');
                return;
            end
            
            relabeler = @f;
            function val=f(val)
                if isKey(renameSet,val)
                    val = renameSet(val);
                end
            end
            curves = Curves.process(curves,relabeler,attr);
        end

        function curves=squeeze(curves,attr,concat)
            if isa(concat, 'function_handle')
                groupSet = containers.Map;
                for ii=1:numel(curves)
                    if isKey(groupSet,curves{ii}.(attr))
                        groupSet(curves{ii}.(attr)) = concat(groupSet(curves{ii}.(attr)),curves{ii});
                    else
                        groupSet(curves{ii}.(attr)) = curves{ii};
                    end
                end
                labels = keys(groupSet);
                curves = cell(length(groupSet),1);
                for ii=1:length(groupSet)
                    curves{ii} = groupSet(labels{ii});
                end
            end
        end
        
        function sortedCurves = sort(curves,accessor)
            values = Curves.aggregate(curves,accessor);
            try
                values = cell2mat(values);
            catch
                1;
            end
            [~,order] = sort(values);
            sortedCurves = cell(numel(curves),1);
            for ii=1:numel(curves); sortedCurves{ii} = curves{order(ii)}; end
        end



        function curves=mean(curves,dim)
            d = 0;
            if nargin>1
                d = dim;
            else
                if curves.repDim ~= 0
                    d = curves.repDim;
                end
            end
            if d == 0
                d = 1;
                disp('Guessing mean is across the 1''st dimension')
            end

            processor = @(cu,cudata) mean(cudata.y,d);
            curves.processY(processor);
        end
        
    
        %% convenience functions
        
        function x = smooth(x,windowSize)
            avgWindow = ones(1,windowSize)/windowSize;
            x = filter(avgWindow,1,x);
        end
        
        function data = fileLoader(path)
            obj = load(path);
            fields = fieldnames(obj);
            data = obj.(fields{1});
        end

        function files = dirLister(path,fullPath)
            if nargin < 2; fullPath = true; end
            files = strsplit(ls(path),{' ','\t','\n'},'CollapseDelimiters',true);
            if strcmp(files{end},''); files = files(1:end-1); end
            if fullPath
                if ~strcmp(path(end),'/'); path = [path '/']; end
                for ii=1:numel(files); files{ii} = [path files{ii}]; end
            end
        end
        
        function r = startsWith(s,prefix) 
            pos = strfind(s,prefix);
            r = (~isempty(pos) && pos==1);
        end

        function r = endsWith(s,suffix) 
            pos=strfind(s,suffix);
            r = (~isempty(pos) && pos+length(suffix)-1==length(s));
        end
        
        function exp = normalizeX(exp,unit)
            exp.x = exp.x * (exp.avgTime/unit);
        end
    end
end

        