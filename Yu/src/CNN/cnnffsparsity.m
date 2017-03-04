function net = cnnffsparsity(net, x)  % y
    n = numel(net.layers);
    inputmaps = 1;
    
    %% ClarkWang 2017.02.16 Lab:TA3
    %net.layers{1}.a{1} = x;
    
    %% ClarkWang 2017.02.15 Lab:TA3
    net.layers{1}.a{1} = x;
    
    for l=2:n
        ndim = size(net.layers{l-1}.a{1});
        minN = min(ndim(1),ndim(2));
        if strcmp(net.layers{l}.type, 'c')
            
          %% ClarkWang 2017.02.16 Lab:TA3   
          current_fft_value = zeros(ndim);
          
          for k=1:inputmaps
              
%             for j = 1 : ndim(3)    %  for each output map
%                 current(:,:,j) = fft2(net.layers{l-1}.a{k}(:,:,j));
%             end

            %% ClarkWang 2017.02.15 Lab:TA3
            %% ClarkWang 2017.02.16 Lab:TA3 
            %current_fft_value = fft2( gpuArray(net.layers{l-1}.a{k}) );
            current_fft_value = fft2( net.layers{l-1}.a{k} );
            
            %% ClarkWang 2017.02.16 Lab:TA3
            template = zeros(ndim);
            
            for i=1:floor(minN/2)
                template(i,          i+1:end-i,:)=1;
                template(ndim(1)+1-i,i+1:end-i,:)=1;
                template(i:end-i+1,i,          :)=1;
                template(i:end-i+1,ndim(2)+1-i,:)=1;
                
                filtered_value = current_fft_value.*template;
                
                %% ClarkWang 2017.02.16 Lab:TA3
                % for to parfor
                for j=1:ndim(3)
                    inverse_fft = real(ifft2(filtered_value(:,:,j)));
                    inverse_fft(find(inverse_fft<1.0e-4))=0;
                    filtered_value(:,:,j)=inverse_fft;
                end
                
                %% ClarkWang 2017.02.16 Home
                %net.layers{l}.a{(k-1)*floor(minN/2)+i}=temp;
                
                %% ClarkWang 2017.02.15 Lab:TA3
                %net.layers{l}.a{(k-1)*floor(minN/2)+i} = double(single(gather(filtered_value)));
                net.layers{l}.a{(k-1)*floor(minN/2)+i} = filtered_value;
                
                template = template*0;
            end
          end
          inputmaps = inputmaps*floor(minN/2);
          % net.layers{l}.outputmaps = inputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            for k=1:inputmaps
                %tic
                %[out, idx] = MaxPooling(net.layers{l-1}.a{k}, [2 2]);
                ratio = 2;
                [pooled_value, idx] = MaxPooling(net.layers{l-1}.a{k}, [ratio ratio]);
                %toc
                
                %% ClarkWang 2017.02.16 Home
                %net.layers{l}.a{k}=out;
                
                %% ClarkWang 2017.02.15 Lab:TA3
                net.layers{l}.a{k}=pooled_value;
            end
        end
    end
    
    

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, size(net.layers{n}.a{j}, 1) * size(net.layers{n}.a{j}, 2), size(net.layers{n}.a{j}, 3))];
    end
    
%     select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
%     ProjX = @(X,k)X .* (abs(X) >= select(sort(abs(X), 'descend'),k));
% 
%     niter = 100; D = net.fv';
%     gamma = 1.6/norm(D)^2; k = floor(size(D,2)*0.1); % varying
%     E = [];
%     Xc = zeros(size(D,2),size(y,1));
%     for i=1:niter
%         R = D*Xc-y';
%         E(end+1,:) = sum(R.^2);
%         Xc = ProjX(Xc - gamma * D'*R, k);
%     end
    
%     for i=1:size(net.fv,1)
%         if nnz(net.fv(i,:))/numel(net.fv(i,:))<0.51   % 25/50
%             net.delete = union(net.delete,i);
%         end
%     end
%     if ~isempty(net.delete)
%         net.ffw(:,net.delete)=0;
%         net.fv(net.delete,:)=0;
%     end
    %  feedforward into output perceptrons
    % net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
end
