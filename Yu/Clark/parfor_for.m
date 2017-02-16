A = zeros(4, 10);
parfor i = 1:4
   for j = 1:10
      if j < 6
         A(i, j) = i + j;
      else
         A(i, j) = pi;
      end
   end
end