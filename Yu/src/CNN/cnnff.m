function net = cnnff(net, x)
    n = numel(net.layers);
    inputmaps = 1;
    net.layers{1}.a{1} = x;
    for l=2:n
        ndim = size(net.layers{l-1}.a{1});
        minN = min(ndim(1),ndim(2));
        if strcmp(net.layers{l}.type, 'c')
          current = zeros(ndim);
          for k=1:inputmaps
            for j = 1 : ndim(3)    %  for each output map  fast fourier transform
                current(:,:,j) = fft2(net.layers{l-1}.a{k}(:,:,j));
            end
            temp = zeros(ndim);
            for i=1:floor(minN/2)
                temp(i,          i+1:end-i,:)=1;
                temp(ndim(1)+1-i,i+1:end-i,:)=1;
                temp(i:end-i+1,i,          :)=1;
                temp(i:end-i+1,ndim(2)+1-i,:)=1;
                temp = current.*temp;
                for j=1:ndim(3)
                    tmp = real(ifft2(temp(:,:,j)));
                    tmp(find(tmp<1.0e-4))=0;
                    temp(:,:,j)=tmp;
                end
                net.layers{l}.a{(k-1)*floor(minN/2)+i}=temp;
                temp = temp*0;
            end
          end
          inputmaps = inputmaps*floor(minN/2);
          % net.layers{l}.outputmaps = inputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            temp = zeros(floor(ndim(1)/2),floor(ndim(2)/2),ndim(3));
            for k=1:inputmaps
%                 z = convn(net.layers{l - 1}.a{k}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
%                 net.layers{l}.a{k} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                for i=1:ndim(3)
                    t1=1;
                    for j1=1:2:ndim(1)
                        t2=1;
                        for j2=1:2:ndim(2)
                            tmp = net.layers{l-1}.a{k}( j1:j1+1,j2:j2+1,i );
                            %tmp = sort(tmp(:));
                            temp(t1,t2,i) = max(tmp(:));
                            t2=t2+1;
                        end
                        t1=t1+1;
                    end
                end
                net.layers{l}.a{k}=temp;
            end
        end
    end
    
%     n = numel(net.layers);
%     net.layers{1}.a{1} = x;
%     inputmaps = 1;
% 
%     for l = 2 : n   %  for each layer
%         if strcmp(net.layers{l}.type, 'c')
%             %  !!below can probably be handled by insane matrix operations
%             for j = 1 : net.layers{l}.outputmaps   %  for each output map
%                 %  create temp output map
%                 z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
%                 for i = 1 : inputmaps   %  for each input map
%                     %  convolve with corresponding kernel and add to temp output map
%                     z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
%                 end
%                 %  add bias, pass through nonlinearity
%                 net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
%             end
%             %  set number of input maps to this layers number of outputmaps
%             inputmaps = net.layers{l}.outputmaps;
%         elseif strcmp(net.layers{l}.type, 's')
%             %  downsample
%             for j = 1 : inputmaps
%                 z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
%                 net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
%             end
%         end
%     end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    for i=1:size(net.fv,1)
        if nnz(net.fv(i,:))/numel(net.fv(i,:))<0.51   % 25/50
            net.delete = union(net.delete,i);
        end
    end
    if ~isempty(net.delete)
        net.ffw(:,net.delete)=0;
        net.fv(net.delete,:)=0;
    end
    %  feedforward into output perceptrons
    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
end
