function [value,vec]=powermethod(A,toler,maxIter)
%
%Power method for computing eigenvalues
%
if nargin<2
    toler=1e-2;
end
if nargin<3
    maxIter=30;
end
[m,n]=size(A);
if m<n
    A=A*A';
else
    A=A'*A;
end
x=randn(size(A,1),1);
dd=1;
ii = 0;
while dd> toler && ii<maxIter
    ii = ii+1;
    y=A*x;
    dd=abs(norm(x)-n);
    n=norm(x);
    x=y/n;
end
vec=x;
vec=vec./norm(vec);

value=sqrt(n);

