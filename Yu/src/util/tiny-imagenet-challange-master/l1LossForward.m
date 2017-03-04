function y = l1LossForward(x,r)

delta = x - r ;
y = sum(abs(delta(:))) ;
y = y / (size(x,1) * size(x,2)) ;  % normalize by image size


