% download a pre-trained CNN from the web (needed once)
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
  'imagenet-googlenet-dag.mat') ;

% setup MatConvNet
run  matlab/vl_setupnn

% load the pre-trained CNN
net = load('imagenet-vgg-f.mat') ;