function imdb = tiny_imagenet_setup_data()

dataDir = fullfile('data', 'tiny-imagenet-200') ;

%
% Find mata data
%
fileWnids = fopen(fullfile(dataDir, 'wnids.txt')) ;
wnids = textscan(fileWnids, '%s') ;
fclose(fileWnids) ;

imdb.classes.names = wnids{1} ;
imdb.imageDir = dataDir ;

% FIXME: add class descriptions

%
% Training images
%
names = {} ;
labels = {} ;
for d = dir(fullfile(dataDir, 'train', 'n*'))'
    [~,lab] = ismember(d.name, imdb.classes.names) ;
    ims = dir(fullfile(dataDir, 'train', d.name, 'images', '*.JPEG')) ;
    names{end+1} = strcat(['train', filesep, d.name, filesep, 'images', filesep], {ims.name}) ;
    labels{end+1} = ones(1, numel(ims)) * lab ;
end
names = horzcat(names{:}) ;
labels = horzcat(labels{:}) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

%
% Validation images
%
fileValAnnotations = fopen(fullfile(dataDir, 'val', 'val_annotations.txt')) ;
val_annotations = textscan(fileWnids, '%s %s %d %d %d %d') ;
fclose(fileValAnnotations) ;

names = {} ;
labels = {} ;
for i=1:length(val_annotations{1})
    [~,lab] = ismember(val_annotations{2}{i}, imdb.classes.names) ;
    names{end+1} = fullfile('val', 'images', val_annotations{1}{i}) ;
    labels{end+1} = lab;
end
labels = horzcat(labels{:}) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + length(imdb.images.id)) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

%
% Test images
%
ims = dir(fullfile(dataDir, 'test', 'images', '*.JPEG')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

names = strcat(['test', filesep, 'images', filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + length(imdb.images.id)) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

%
% Load the data
%
data = [] ;

%h = waitbar(0,'Loading data...') ;
progress = 1 ;
for name = imdb.images.name

  %waitbar(progress/length(imdb.images.name), h , sprintf('%.2f%% along...', 100 * progress/length(imdb.images.name))) ;
  disp(progress) ;
  path = fullfile(dataDir, name) ;
  im = imread(char(path)) ;
  im = single(im) ;
  
  if (size(im, 3) == 1)
    data = cat(4, data, cat(3, im, im, im)) ; % is this correct?
  else
    data = cat(4, data, im);
  end
  
  progress = progress + 1 ;
end
%close(h) ;

imdb.images.data = data ;
