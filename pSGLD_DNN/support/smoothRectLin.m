function x=smoothRectLin(x)
    lim=10;
    ndx1=x>lim;
    ndx2=x<-lim;
    tm=(~ndx1)&(~ndx2);
    x(tm)=log(1+exp(x(tm)));
    x(ndx2)=exp(x(ndx2));
end