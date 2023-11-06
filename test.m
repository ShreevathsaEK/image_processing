% Load and preprocess a new image (adjust as needed)
newImage = imread("DEER3.jpg");    %add your test image here
newImage = imresize(newImage, [32, 32]);
newImage = double(newImage) / 255;


% prediction
predictedLabel = classify(net, newImage);

% Create a custom figure to display the image
fig = figure('Name', 'Prediction Result', 'NumberTitle', 'off');
imshow(newImage);
title(sprintf('Predicted Label: %s', char(predictedLabel)));


% Wait for the user to close the figure
uiwait(fig);
