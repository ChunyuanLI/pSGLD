function [U,L,V]=decrandsvd(A,k)
% [U,L,V]=decrandsvd(A,k)
% A is m by n
[m,n]=size(A);
if strcmp(class(A),'gpuArray')
    omeg=gpuArray.randn(n,k);
else
    omeg=randn(n,k);
end
Y=A*omeg;
[Q,R]=qr(Y,0);
B=Q'*A;
B2=B*B';
try
[Uhat,L,~]=svd(B2);
catch
    1;
end
L=sqrt(diag(L));
V=bsxfun(@times,B'*Uhat,(L)'.^-1);
% [Uhat, L, V] = svd(B,'econ');
U=Q*Uhat;
% L=diag(L);
return
%%
tic
[Uhat, L, V] = svd(B,'econ');
toc
%%
tic
B2=B*B';
[Uhat2,L2,~]=svd(B2);
L2=sqrt(diag(L2));
V2=bsxfun(@times,B'*Uhat2,(L2)'.^-1);
toc
%%
% V3=bsxfun(@times,B'*Uhat2,diag(L2)'.^-.5);
% plot(V(:,1),V3(:,1),'.')
