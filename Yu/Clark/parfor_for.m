% 
% tic
% clear A
% for i = 1:80000000
%    A(i) = i;
% end
% 
% toc
% 
% tic
% clear B
% parfor i = 1:80000000
%    B(i) = i;
% end
% 
% toc




clear C
d = 0; i = 0;
for i = 1:4
   d = i*2;
   C(i) = d;
end
C
d
i

clear D
d = 0; i = 0;
parfor i = 1:4
   d = i*2;
   D(i) = d;
end
D
d
i




















