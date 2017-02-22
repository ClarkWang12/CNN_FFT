setup('useGpu', true);

imdb = load('data/imdb.mat');
imdb = imdb.imdb;

net = initializeCnn() ;
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/first-test' ;
trainOpts.gpus = [1] ;
trainOpts.batchSize = 8 ;
trainOpts.learningRate = 0.0001 ;
trainOpts.plotDiagnostics = false ;
trainOpts.numEpochs = 20 ;
trainOpts.errorFunction = 'multiclass' ;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;