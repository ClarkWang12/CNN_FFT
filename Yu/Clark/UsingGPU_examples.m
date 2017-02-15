x=4;
array_size= 1024;
data = rand(array_size,array_size);

%% CPU
D = data;
for iii = 1:x;
tic
    for i = 1:10.^(iii-1); % iterations
    Y = (fft2(D)); 
    end
t_cpu(iii) = toc;
end

%% GPU
D = gpuArray(data);
for iii = 1:x
tic
parfor i = 1:10.^(iii-1) % iterations
    Y = gather(fft2(D)); 
end 
    t_pgpu(iii) = toc;
end