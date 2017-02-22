%bounding box
In = imread('Input4image.jpg');
a = In;
figure, imshow(a), title('original');
b = rgb2gray(a);
%figure, imshow(b), title('grayscale');
c = 255-b;
%figure, imshow(c), title('complemented');
d = im2bw(c);
%figure, imshow(d), title('black and white');
e = imfill(d,'holes');
%figure, imshow(e), title('filled');
f = bwlabel(e);
%figure, imshow(f), title('labelled bw');
vislabels(f),title('labelled');
g = regionprops(f,'BoundingBox'); %'Area'
area_image = regionprops(f,'Area');
%bb = ones(4);
%g(1)
%bb = g(1).BoundingBox;
%disp('bb');
%disp(bb);
%area_values = [g.Area]
%idx = find((4000 <= area_values) & (area_values <= 6000))
%h = ismember(f,idx);

bbvalues = zeros(4);

count = 0;
ara = size(a);
totarea = ara(1)*ara(2);
for k =1:length(g)
    a1 = area_image(k).Area;
   if a1 > 300 %0.1*totarea
       count = count + 1;
    bb = g(k).BoundingBox;
    bbvalues(count,:) = bb;
    
    rectangle('Position', [bb(1), bb(2), bb(3),bb(4)],...
    'EdgeColor', 'r', 'LineWidth',2)

inp = In;
input = imcrop(inp,bb);
%imresize(input,0.1);
imwrite(input, 'input.jpg', 'jpg')
%imshow('input.jpg');

im = imread('input.jpg') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;
objname{count} =  net.classes.description{best};
    end;
end;

s = size(bbvalues);
limit= s(1);
prep=[limit,limit];
bl = blanks(1);
for i=1:limit
  
    for j= 1:limit
        if i~=j
            i1 = bbvalues(i,:);
            i2 = bbvalues(j,:);
            x1 = i1(1);
            y1 = i1(2);
            x1_width = i1(3);
            y1_width = i1(4);
            x2 = i2(1);
            y2 = i2(2);
            x2_width = i2(3);
            y2_width = i2(4);
            
           
            if x1+x1_width<x2 
                prep(i,j) = 1;  %left
                
            end; 
            if x1>x2+x2_width
                prep(i,j) = 2;  %right
              
            end;
            if y1+y1_width<y2 
                prep(i,j) = 3;  %above
               
            end;
            if y1>y2 +y2_width
                prep(i,j) = 4;  %below
               
            end;
        end;     
           
          if i==j
            prep(i,j) =0;
       
        end;
    end;
end;

for i =1: limit
    for j=1:limit
        if(i == 2 || i == 3)
        if prep(i,j)==1
            opt = [objname{i},' is to the left of', bl ,objname{j}];
            disp(opt);
            op = opt;
            opt1 = ['say', blanks(1), opt];
            system(opt1);
        end;
         if prep(i,j)==4
            opt = [objname{i},' is below ',bl, objname{j}];
            disp(opt);
            opt1 = ['say', blanks(1), opt];
            system(opt1);
            
        end;
        end;
        
        if(i == 1 || i == 4)
        if prep(i,j)==2
            opt = [objname{i},' is to the right of ',bl,objname{j}];
            disp(opt);
            opt1 = ['say', blanks(1), opt];
            system(opt1);
        end;
        if prep(i,j)==3
            opt = [objname{i},' is above ', bl, objname{j}];
            disp(opt);
            opt1 = ['say', blanks(1), opt];
            system(opt1);
        end;
        end;
       
    end;
end;

%image search
url = 'http://images.google.com/images?&q=';
url =strcat(url,op);
web(url);
