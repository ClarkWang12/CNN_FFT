A1 = zeros(10,10); 

parfor ix = 1:10
    for jx = 1:10
        A1(ix, jx) = ix + jx;
    end
end