function rdata =  ToyData(N)
J=10;
M=200;
sig2=.5;
persistent data;
data = {};
data.inSize = M;
data.outSize = J;
data.train = {};
data.test = {};
x=randn(M,N);
data.train.inputs=x;
W=sqrt(sig2)*randn(J,M);
py=softmax(W*x);
y=mnrnd(1,py')';
[~,r]=max(y);
data.train.labels=r';
x=randn(M,N);
data.test.inputs=x;
%     W=sqrt(sig2)*randn(J,M);
py=softmax(W*x);
y=mnrnd(1,py')';
[~,r]=max(y);
data.test.labels=r';

rdata = data;
end
