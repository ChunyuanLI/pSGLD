function [kernels] = projectKernels(kernels,gpu, proj,k)
    sz = size(kernels);
    for ii=1:sz(3)
        for jj=1:sz(4)
            if any(any(kernels(:,:,ii,jj)))
                kernels(:,:,ii,jj) = reshape(projectVector(reshape(kernels(:,:,ii,jj),[prod(sz(1:2)) 1]),gpu,proj,k),sz(1:2));   
            end
        end
    end
   