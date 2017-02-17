close all, clear all,
addpath(genpath('./DeepLearnToolbox-master'));

load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;   % 60000 images for training
test_x = double(reshape(test_x',28,28,10000))/255;     % 10000 images for test
train_y = double(train_y');                            % for training set, each input with 10 features and total labels are 60000
test_y = double(test_y');                              % for test set,

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network % Alex Minnaar Chris McCormick Andrew Gibiansky
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
cnn = cnnsetup(cnn, train_x, train_y);
% 10 labels correspond to 0-9 digits, and the final maps with 4*4*12 features per map to fully connected network
opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 1;



tic
cnn = cnntrainsparsity(cnn, train_x, train_y, opts); % cnntrain(cnn, train_x, train_y, opts);
toc

save('cnntrain_Result.mat', 'cnn');

tic
[er, bad] = cnnsparsityexamples(cnn, test_x, test_y); % cnntest(cnn, test_x, test_y);
toc

er

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');