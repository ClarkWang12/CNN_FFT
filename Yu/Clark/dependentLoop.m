% Example of a dependent for-loop
a = zeros(1,10);

parfor it = 1:10 
    a(it) = someFunction(a(it-1));
end