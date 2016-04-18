function [h, img] = visStuff(A, imgsize, opts)

if nargin<3;opts = {};end
if ~isfield(opts,'normalize'); opts.normalize=true; end
if ~isfield(opts,'rotate'); opts.rotate=0; end

while imgsize(end)==1; imgsize = imgsize(1:end-1); end %get rid of trailing singleton dimensions

warning off all

A = squeeze(A);
dData = ndims(A);
dShape = numel(imgsize);

if dShape == 3
    switch(dData)
        case 4
            flat = false;
            n = size(A,4);
        case 3
            flat = false;
            n = 1;
        case 2
            flat = true;
            n = size(A,2);
        case 1
            flat = true;
            n = 1;
    end

elseif dShape == 2
    switch(dData)
        case 3
            flat = false;
            n = size(A,3);
        case 2
            if all(size(A)==imgsize)
                flat = false;
                n = 1;
            else
                flat = true;
                n = size(A,2);
            end
        case 1
            flat = true;
            n = 1;
    end
end
    

if opts.normalize
    A = A - min(A(:));
    A = A/max(A(:));
end

if flat
    A = reshape(A,[imgsize n]);
end

if n==1
    img = A;
else
    m = ceil(sqrt(n));
    sepWidth = 2;
    sepCol = 1;
    horSeparatorSize = [sepWidth m*(imgsize(2)+sepWidth)+sepWidth ];
    verSeparatorSize = [imgsize(1) sepWidth];
    if dShape==3
        horSeparatorSize = [horSeparatorSize imgsize(3)]; 
        verSeparatorSize = [verSeparatorSize imgsize(3)]; 
    end
    horSeparator = sepCol*ones(horSeparatorSize);
    verSeparator = sepCol*ones(verSeparatorSize);
    img = [horSeparator];
    indexer(1:dShape) = {':'};
    indexer(dShape+1) = {1};
    
    for ii=1:m
        row = verSeparator;
        for jj=1:m
            if indexer{end} > n
                row = cat(2, row, sepCol*ones(imgsize), verSeparator); 
            else
                row = cat(2, row, A(indexer{:}), verSeparator); 
                indexer{end} = indexer{end} + 1;
            end
        end
        img = cat(1, img, row, horSeparator);
    end
end

if opts.rotate ~= 0
    img = rot90(img,opts.rotate);
end

figure;
h = imshow(img);
%h = imagesc(img);
axis image off
drawnow;
warning on all
