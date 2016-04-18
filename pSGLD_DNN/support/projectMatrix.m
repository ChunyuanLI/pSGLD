function x=projectMatrix(x,gpu,projectionName,xnorm)
if nargin<3;projectionName='spectral';end
if nargin<4
    switch projectionName
        case 'linear'
            x=x;
        case 'spectral'
            x=projectSpectral(x);
        case 'spectraltrue'
            x=projectSpectralTrue(x,gpu);
        case 'approxSpectral'
            x=projectapproxSpectral(x,gpu);
        case 'approxRandSpectral'
            k=10;
            x=projectApproxRandSpectral(x,gpu);
        case 'approxRandSpectral2'
            x=projectRandSpectral(x,gpu);
        case 'semispectral'
            x=projectSemiSpectral(x);
        case 'l_inf'
            x=projectLInfinity(x);
        case 'nuclear'
            x=projectNuclear(x);
        case 'L1row'
            x=projectL1row(x,gpu);
        case 'L2row'
            x=projectL2row(x);
        otherwise
            error('Not a used norm.')
    end
else
    switch projectionName
        case 'linear'
            x=x;
        case 'spectral'
            x=projectSpectral(x,xnorm);
        case 'spectraltrue'
            x=projectSpectralTrue(x,gpu,xnorm);
        case 'semispectral'
            x=projectSemiSpectral(x,xnorm);
        case 'approxRandSpectral'
           
            x=projectApproxRandSpectral(x,gpu,xnorm);
        case 'l_inf'
            x=projectLInfinity(x,xnorm);
        case 'nuclear'
            x=projectNuclear(x,xnorm);
        otherwise
            error('Not a used norm.')
    end
end

function P=projectApproxRandSpectral(x, ~, k)
%%
[U,L,V]=decrandsvd(x,k);
x2=x-U*diag(L)*V';
L2=powermethod(x2);
Lall=sum(L)+L2;
P=Lall*U*V'+Lall./L2*x2;
1;



%%

function P=projectapproxSpectral(x, ~, ~)
%%
[m,n]=size(x);
if m<n
    [val1,lV]=powermethod(x,1e-2);
    rV=lV'*x./val1;
else
    [val1,rV]=powermethod(x,1e-2);
    lV=x*rV./val1;rV=rV';
end
1;
x1=x-val1*lV*rV;
val2=powermethod(x1,1e-3);
P=(val1+val2)*(x1/val2+lV*rV);
1;
if sum(isnan(P(:)))
    1;
end


function P=projectL1row(x, gpu, ~)
rows=1:size(x,1);
[mvals,idx] = max(abs(x),[],2);
idx = sub2ind(size(x),rows(:),idx(:));
P = zeros(size(x));
if gpu;P=gpuArray(P);end
P(idx) = sum(mvals)*sign(x(idx));

function y=projectL2row(x,~)
s=norm(x,'fro');
y=s*bsxfun(@rdivide,x,sqrt(sum(x.^2,2)+1e-14));
1;

function x=projectSpectral(x,mS)
[U,S,V]=svd(x,'econ');
if nargin<2
    mS=S(1);
end
S=diag(S);
r=sum(S>mS*1e-8);
x=U(:,1:r)*V(:,1:r)'*mS;

function x=projectSpectralTrue(x,gpu,mS)
[U,S,V]=svd(x,'econ');
S=diag(S);
if nargin<3
    mS=sum(S);
end
r=sum(S>mS*1e-8);
if gpu
    one2r = gpuArray.colon(1, r);
    x=U(:,one2r)*V(:,one2r)'*mS;
else
    x=U(:,1:r)*V(:,1:r)'*mS;
end

function x=projectRandSpectral(x,gpu,mS)
[U,S,V]=randomizedSVD(x, min(size(x)), [] , 1);
S=diag(S);
if nargin<3
    mS=sum(S);
end
r=sum(S>mS*1e-8);
if gpu
    one2r = gpuArray.colon(1, r);
    x=U(:,one2r)*V(:,one2r)'*mS;
else
    x=U(:,1:r)*V(:,1:r)'*mS;
end

function x=projectSemiSpectral(x,r)
if nargin<2
    r=min(2,min(size(x)));
end
[U,S,V]=svd(x,'econ');
x=U(:,1:r)*V(:,1:r)'*S(1);


function x=projectLInfinity(x,mS)
[M,J]=size(x);
if nargin<2
    mS=max(sum(abs(x),2));
end
for m=1:M
    x(m,:)=x(m,:).*(mS./sum(abs(x(m,:))));
end

function x=projectNuclear(x,mS)
[V,D]=eigs(x'*x,1);
x=x*V*V';
if nargin>=1
    x=x.*mS./norm(x,'fro');
end