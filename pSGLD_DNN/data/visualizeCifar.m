function [h, array] = displayCifar(A, opt_normalize, opt_graycolor, cols, opt_colmajor)
% This function visualizes filters in matrix A. Each column of A is a
% filter. We will reshape each column into a square image and visualizes
% on each cell of the visualization panel. 
% All other parameters are optional, usually you do not need to worry
% about it.
% opt_normalize: whether we need to normalize the filter so that all of
% them can have similar contrast. Default value is true.
% opt_graycolor: whether we use gray as the heat map. Default is true.
% cols: how many columns are there in the display. Default value is the
% squareroot of the number of columns in A.
% opt_colmajor: you can switch convention to row major for A. In that
% case, each row of A is a filter. Default value is false.
warning off all

if ~exist('opt_normalize', 'var') || isempty(opt_normalize)
    opt_normalize= true;
end

if ~exist('opt_graycolor', 'var') || isempty(opt_graycolor)
    opt_graycolor= true;
end

if ~exist('opt_colmajor', 'var') || isempty(opt_colmajor)
    opt_colmajor = false;
end

% rescale
%A = A - mean(A(:));

if opt_graycolor, colormap(gray); end

% compute rows, cols
[M, L]=size(A);
sz=floor(sqrt(L));
buf=1;
if ~exist('cols', 'var')
    if floor(sqrt(M/3))^2 ~= M
        n=ceil(sqrt(M/3));
        while mod(M/3, n)~=0 && n<1.2*sqrt(M/3), n=n+1; end
        m=ceil(M/(3*n));
    else
        n=sqrt(M/3);
        m=n;
    end
else
    n = cols;
    m = ceil(M/n);
end

A = A - min(A(:));
A = A/max(A(:));

ind = 1;
sepWidth = 2;
sepCol = 1;
nn = 1; 
array = [sepCol*ones(sepWidth, sz*(n+sepWidth)+sepWidth, 3)]; %3 for RGB
for ii=1:sz
    row = [sepCol*ones(n,sepWidth,3)];
    for jj=1:sz
        if opt_normalize
            %nn = max(abs(A(:,ind)));
        else
            %nn = max(abs(A(:)));
        end
        row = [row reshape(A(:,ind),[n,n,3])/nn sepCol*ones(n,sepWidth,3)]; 
        ind = ind + 1;
    end
    array = [array ; row ; sepCol*ones(sepWidth, sz*(n+sepWidth)+sepWidth, 3)];
end

figure;
h = imshow(rot90(array,-1));
axis image off

drawnow;

warning on all
