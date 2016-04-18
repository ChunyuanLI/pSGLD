clearvars;
%clc;

format shortG;

rng(123,'twister')

n=100;
opts.padding = false;
opts.gpu = true;
opts.precision = 'single';

c1 = SpatialConvolution(3,4,3,3,opts);
c1.tightPadding();

p1 = Pooling(2,2);
p1.noOverlap();

c2 = SpatialConvolution(4,3,2,2,opts);
c2.tightPadding();

c2outSize = prod(c2.getOutputSize(p1.getOutputSize(c1.getOutputSize([6 6]))));

nn = Sequential(opts);
nn.add(c1);
nn.add(p1);
nn.add(Sigmoid());
nn.add(c2);
nn.add(Flatten(n==1));
%nn.add(Linear(10,5));
%nn.add(Sigmoid());
nn.add(Linear(c2outSize*c2.outChannels,1));

%Solve XOR problem

%inputs = [0 0 ; 0 1; 1 0; 1 1]';
%targets = [0; 1; 1; 0]';

input = rand(6,6,3,n,opts.precision);
target = rand(1,n,opts.precision);

if opts.gpu
    input = gpuArray(input);
    target = gpuArray(target);
end

c = MSECriterion();

maxIter = 0;
alpha = 0.2;

errs = [];

for it=1:maxIter
    err = 0;
    for i=1:size(inputs,2)
        input = inputs(:,i);
        target = targets(:,i);
        output = nn.forward(input);
        err = err + c.forward(output,target);
        %disp('---------')
        grad = c.backward(output,target);
        nn.backward(input,grad);
    end
    errs = [errs err/size(inputs,2)];
    nn.updateParameters(alpha,1,true);
end

plot(errs)


%Gradient checking
%input = inputs(:,1);
%target = targets(:,1);
params = nn.getParameters();
epsilon = 1e-4;
gradApprox = zeros(length(params),1);
if opts.gpu
    gradApprox = gpuArray(gradApprox);
end
for i=1:length(params)
    pertrub = zeros(length(params),1);
    pertrub(i) = epsilon;
    
    nn.setParameters(params + pertrub);
    output = nn.forward(input);
    plusCost =  c.forward(output,target);
    
    nn.setParameters(params - pertrub);
    output = nn.forward(input);
    minusCost =  c.forward(output,target);
    
    gradApprox(i) = (plusCost - minusCost)/(2*epsilon);
end

nn.setParameters(params);
output = nn.forward(input);
grad = c.backward(output,target);
nn.backward(input,grad);
grad = nn.getParametersGradient();

res = [grad  gradApprox grad-gradApprox]
max(res(:,3))













