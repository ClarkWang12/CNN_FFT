function dx = l1LossBackward(x,r,p)

dx = p * sign(x-r);
dx = dx / (size(x,1) * size(x,2)) ;  % normalize by image size
