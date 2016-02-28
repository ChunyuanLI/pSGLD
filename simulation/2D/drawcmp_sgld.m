clear all;
load pre_sgld_cmpdata;

for i = 1 : nset
  SGLDCovErr(i)   = sum(sum(abs( covESGLD(:,:,i) - meanESGLD(:,i)*meanESGLD(:,i)' - covS ))) / 4;
  rmsprop_SGLDCovErr(i)   = sum(sum(abs( rmsprop_covESGLD(:,:,i) - rmsprop_meanESGLD(:,i)*rmsprop_meanESGLD(:,i)' - covS ))) / 4;
end

len = 4; str = {'1','2','3','4','5'};
figure(); 
plot( SGLDauc, SGLDCovErr , 'r-o' ); hold on;
for i=1:5, text( SGLDauc(i), SGLDCovErr(i), str{i},'HorizontalAlignment','right', 'Fontsize', 15, 'color','red'); end
plot( rmsprop_SGLDauc,  rmsprop_SGLDCovErr, 'b-o' ); hold on;
for i=1:5, text( rmsprop_SGLDauc(i), rmsprop_SGLDCovErr(i), str{i},'HorizontalAlignment','right', 'Fontsize', 15, 'color','blue'); end

xlabel('Autocorrelation Time');
ylabel('Average Absolute Error of Sample Covariance');

legend( 'SGLD', 'pSGLD');

fgname = 'figure/sgldcmp-preconditioner';

set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )

saveas( gcf, fgname, 'fig');
saveas( gcf, fgname, 'pdf');
