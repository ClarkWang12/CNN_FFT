
dataDir = '/media/clark/ClarkWang/Data/ImageNet';

lite = false;

root = fileparts(fileparts(mfilename('fullpath'))) ;

imdb = cnn_imagenet_setup_data('dataDir', dataDir, 'lite', lite) ;
