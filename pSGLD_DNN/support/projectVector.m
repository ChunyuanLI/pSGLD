function x=projectVector(x,gpu,projectionName,projectionLimit)
if nargin<3;projectionName='linear';end
if nargin<4
switch projectionName
    case 'linear'
        x=x;
    case 'infinity'
        x=sign(x)*max(abs(x));
    case 'infinitytrue'
        x=sign(x)*sum(abs(x));
    case 'l1'
        [m,ndx]=max(abs(x));
        m=m*sign(x(ndx));
        x=zeros(size(x));
        if gpu;x=gpuArray(x);end
        x(ndx)=m;
end
else
switch projectionName
    case 'linear'
        x=x;
        x=x./norm(x,'fro')*projectionLimit;
    case 'infinity'
        x=sign(x)*projectionLimit;
    case 'l1'
        [m,ndx]=max(abs(x));
        m=m*sign(x(ndx));
        x=zeros(size(x));
        if gpu;x=gpuArray(x);end
        x(ndx)=sign(M)*projectionLimit;
end
end