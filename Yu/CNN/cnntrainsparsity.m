function net = cnntrainsparsity(net, x, y, opts)
    m = size(x, 3);                  % sample number
    numbatches = m / opts.batchsize; % 1200 batch number
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        kk = randperm(m);
        for l = 1 : numbatches
            %disp(l)
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));

            net = cnnffsparsity(net, batch_x); %, batch_y);
            
            if ~isfield(net,'dict')
                net.dict = zeros(size(net.fv,1),size(batch_y,1));
            end
            
            %disp('fullyconnectionsparsity')
            fullyconnectionsparsity;
            
%             net = cnnbp(net, batch_y);
%             net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            else
                net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            end
            
        end
    end
    

    function fullyconnectionsparsity
        select = @(A,k) repmat(A(k,:), [size(A,1) 1]);
        ProjX = @(X,k) X .* (abs(X) >= select(sort(abs(X), 'descend'),k));

        niter = 100; D = net.fv'; Yval=batch_y';
        gamma = 1.6/norm(D)^2; ksparse = floor(size(D,2)*0.5); % varying
        E = [];    Xc = net.dict;                     % Xc = zeros(size(D,2),size(Yval,2));
        for it=1:niter
            R = D*Xc-Yval;
            E(:,end+1) = sum(R.^2,2);
            Xc = ProjX(Xc - gamma * D'*R, ksparse);
        end
        net.dict = Xc;
        net.L = 1/2* sum(E(:)) / size(E, 1);
%        net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
    end
end