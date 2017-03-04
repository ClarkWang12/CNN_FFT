train_folderPath = '\train\';
test_folderPath = '\Testing\*.jpg';
test_folder = 'Testing\';
D = dir(['\train\*']);
num_classes=0;
classes=[];
str2 ='.';
for i=1:length(D)
    str = D(i).name;
    x=findstr(str,str2);
    if(size(x)<1)
        num_classes = num_classes+1;
        classes{num_classes}=str;
    end;
end;
classes = lower(classes)
%disp(classes);
p=0;
a=[];
imgdata=[];
count_class=[];
classlabels = []
for i=1: length(classes)
    
    foldername  = classes(i);
    folderPath = strcat(train_folderPath,foldername);
    format ='\*.jpg';
    folder = char(folderPath);
    folderName = strcat(folder,format);
    srcFiles  = dir([folderName]);
    count_class{i} = length(srcFiles);
   % if(i>1)
    %    for m =1: (i-1)
    %    p = p + count_class{m};
    %    end;
    %end;
    
    for j=1:length(srcFiles)
        p = p+1;
        to_read = strcat(folder,'\',srcFiles(j).name);
        im = imread(to_read);                           %read image
        %disp(to_read)
        [rows columns numberOfColorChannels] = size(im);
        if numberOfColorChannels == 3
        im = rgb2gray(im);                      %rgb to gray
        %sceneImage = imadjust(sceneImage);
        end
        sceneImage = histeq(im);
        
        sceneImage = imadjust(sceneImage,stretchlim(sceneImage),[]);
         B = imresize(sceneImage, [50 50]);
         B1 = B';
         B2 = B1(:)';
         a = [a;B2];
        % classlabels(p) = classes(i);
        classlabels = [classlabels,i]
        %classlabels(p) = i;
       end;
   end;
   classlabels = transpose(classlabels);
knn = fitcknn(a,classlabels)
%-----------
T=[];

label_index=0;
%num_test =8;
testlabels = []
disp('READING TESTING');
 testFiles  = dir([test_folderPath]);
 
  for j=1:length(testFiles)
      label_index = label_index +1;
        to_read = strcat(test_folder,testFiles(j).name);
        %disp(to_read);
        im = imread(to_read);                           %read image
        [rows columns numberOfColorChannels] = size(im);
        if numberOfColorChannels == 3
        im = rgb2gray(im);                      %rgb to gray
        %sceneImage = imadjust(sceneImage);
        end
        sceneImage = histeq(im);
        %sceneImage = brighten(sceneImage)
         test = imresize(sceneImage, [50 50]);
         T1 = test';
         T2 = T1(:)';
         T = [T;T2];
        % classlabels(p) = classes(i);
        label = testFiles(j).name;
        Ck = strsplit(label,'-')
        vq = lower(Ck{1})
        x = strmatch(vq, classes)
        y = isempty(x)
        if y ==1
           x = y-1
        end
        testlabels = [testlabels;x]
        label = predict(knn,T2)
        %testlabels(j) = x
  end;
testlabels = transpose(testlabels)
labelknn = predict(knn,T)
disp('detected labels: ')
for i = 1:length(labelknn)
      disp(classes{labelknn(i)})
end
count = 0
labelknn(i) = labelknn(i)'
for i = 1:length(labelknn)
	if(labelknn(i) == testlabels(i))
		count = count + 1
	end
end
accuracy = count/(length(labelknn))
disp(accuracy)
disp('-----------------------------------')
disp('Decision Tree')
a = double(a)
naive = fitctree(a,classlabels)
T = double(T)
labeldt = predict(naive,T)
disp('detected labels: ')
for i = 1:length(labeldt)
      disp(classes{labeldt(i)})
end
count = 0
for i = 1:length(labeldt)
	if(labeldt(i) == testlabels(i))
		count = count + 1
	end
end
accuracydecisiontree = count/(length(labeldt))

%----------------------------------------
disp('Support Vector Machines')
Group = multisvm(a,classlabels,T)
[m,n] = size(Group);
count = 0;
for i = 1:m
	if(Group(i) == testlabels(i))
		count = count + 1
	end
end
accuracySVM = (count)/m;

disp('-----------------------------------')
disp('Naive Bayes')
naive = fitcnb(a,classlabels)
labelnaive = predict(naive,T)
disp('detected labels: ')
for i = 1:length(labelnaive)
      disp(classes{labelnaive(i)})
end
count = 0
for i = 1:length(labelnaive)
	if(labelnaive(i) == testlabels(i))
		count = count + 1
	end
end
accuracynaivebayes = count/(length(labelnaive))
%-------------------------------------------
dff = []
dff = [a classlables dff]
dff = dff(randperm(2322),:)
 train = dff(1:1858,:);
test = dff(1859:2322,:);
training_1 = train(:,1:2500)
training_labels = train(:,2501)
 test_1 = test(:,1:2500)
test_labels = test(:,2501)
