tic
s=0;
for i=1:100000
    s=s+(1/2^i+1/3^i); 
end
s
toc


tic
i=1:100000;
s=sum(1./2.^i+1./3.^i)
toc


% Example
%     -------
%     Calculate the local mean using a [2 2] neighborhood with zero padding.
%  
%         A = reshape(linspace(0,1,16),[4 4])'
%         B = im2col(A,[2 2])
%         M = mean(B)
%         newA = col2im(M,[1 1],[3 3])