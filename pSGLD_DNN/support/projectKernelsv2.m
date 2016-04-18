function [kernels] = projectKernelsv2(kernels,gpu, proj,k)
sz = size(kernels);
X=reshape(kernels,prod(sz(1:3)),sz(4));
X=projectMatrix(X,gpu,proj,k);
kernels=reshape(X,sz);
% for ii=1:sz(3)
%     for jj=1:sz(4)
%         if any(any(kernels(:,:,ii,jj)))
%             kernels(:,:,ii,jj) = projectMatrix(kernels(:,:,ii,jj),gpu,proj,k);
%         end
%     end
% end
