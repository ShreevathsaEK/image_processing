% Download CIFAR-10 dataset if not already downloaded
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
downloadFolder = tempdir;
filename = fullfile(downloadFolder, 'cifar-10-matlab.tar.gz');

dataFolder = fullfile(downloadFolder, 'cifar-10-batches-mat');
if ~exist(dataFolder, 'dir')
    fprintf('Downloading CIFAR-10 dataset (175 MB)... ');
    websave(filename, url);
    untar(filename, downloadFolder);
    fprintf('Done.\n');
end

% Load CIFAR-10 dataset
[XTrain, YTrain, XTest, YTest] = loadCIFARData(downloadFolder);  
XTrain = double(XTrain) / 255; 
XTest = double(XTest) / 255;

% defining the layers using cnn
layers = [
    imageInputLayer([32 32 3]) 
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()
];

% defining the options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% Train the network
net = trainNetwork(XTrain, YTrain, layers, options);
YTestPredicted = classify(net, XTest);
accuracy = sum(YTestPredicted == YTest) / numel(YTest) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);
