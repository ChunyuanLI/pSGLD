%% This file produces Figure 1a, running trace of SGLD and pSGLD
% revised based on the code of SGHMC by Tianqi Chen

clear all;
global covS;
global invS;

V = 1;

% covariance matrix
rho = 0;
covS = [ 0.16, rho; rho, 1 ];
invS = inv( covS );

% intial 
x = [0;0];

% this is highest value tried so far for SGLD that does not diverge
etaSGLD = 0.3;
etaSGHMC = 0.05;
alpha = 0.035;

% number of steps 
L = 600;

probUMap = @(X,Y) exp( - 0.5 *( X .* X * invS(1,1) + 2 * X.*Y*invS(1,2) + Y.* Y *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));   
funcU = @(x) 0.5 * x'*invS*x;
gradUTrue = @(x) invS * x;
gradUNoise = @(x) invS * x  + 0.1*randn(2,1);


% set random seed
randn( 'seed',20 );

dsgld = sgld( gradUNoise, etaSGLD, L, x, V );
pre_dsgld = sgld_rmsprop( gradUNoise, etaSGLD, L, x, V );



% plot the figures
figure(); axes('FontSize', 15);
subplot(1,2,1);
[XX,YY] = meshgrid( linspace(-2,2), linspace(-2,2) );
ZZ = probUMap( XX, YY ); contour( XX, YY, ZZ, 'LineWidth', 2); hold on;
h1=plot( dsgld(1,:), dsgld(2,:),'.',...
    'MarkerSize',8,...
    'MarkerEdgeColor',[0.7,0.7,0.7],...
    'MarkerFaceColor',[0.8,0.8,0.8]);

%for i=1:length(dsgld), text( dsgld(1,i), dsgld(2,i), num2str(i),'HorizontalAlignment','right', 'Fontsize', 10, 'color','red'); end
xlabel('x'); ylabel('y'); legend(h1, 'SGLD', 'fontsize', 10); axis([-4.1 4 -4.1 4]); len = 5;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )


subplot(1,2,2);
[XX,YY] = meshgrid( linspace(-2,2), linspace(-2,2) );
ZZ = probUMap( XX, YY ); contour( XX, YY, ZZ,'LineWidth', 2 ); hold on;
h3=plot( pre_dsgld(1,:), pre_dsgld(2,:),'.',...
    'MarkerSize',8,...
    'MarkerEdgeColor',[0.7,0.7,0.7],...
    'MarkerFaceColor',[0.8,0.8,0.8]);

%for i=1:length(dsgld), text( pre_dsgld(1,i), pre_dsgld(2,i), num2str(i),'HorizontalAlignment','right', 'Fontsize', 10, 'color','red'); end
xlabel('x'); ylabel('y'); legend(h3, 'pSGLD', 'fontsize', 10); axis([-4.1 4 -4.1 4]); len = 5;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )




fgname = 'figure/sgldcmp-run';
saveas( gcf, fgname, 'fig');
saveas( gcf, fgname, 'pdf');

