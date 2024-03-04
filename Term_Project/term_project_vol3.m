% Clear workspace and load data
clear
clc

load('term_project.mat');

% Check if neural network exists, if not, train it
netExists = exist('net', 'var');
if ~netExists
    % Preprocess training images
    trainImgNames = ["5tl.jpg", "10tl.jpg", "20tl.jpg", "50tl.jpg", "100tl.jpg", "200tl.jpg"];
    trainImages = preprocessImages(trainImgNames);

    % Define target outputs for training
    output = eye(6);

    % Create and train neural network
    net = createAndTrainNetwork(trainImages, output);

    % Display training results
    displayTrainingResults(net, trainImages, output);
end

% Classify test image
[testImg, actualClass] = preprocessTestImage('test_20tl.jpg');

% Display the processed test image for debugging
figure()
imshow(reshape(testImg, [62 107]), "InitialMagnification",500);
title('Processed Test Image');

% Perform prediction using the trained network
prediction = net(testImg);
fprintf('Raw Prediction Values:\n');
disp(prediction);

% Find the predicted class and its associated probability
[~, predictedClass] = max(prediction);
fprintf('Predicted class: %s with probability %.4f\n', trainImgNames(predictedClass), prediction(predictedClass));

% Display the actual probabilities as percentages
disp('Actual Probabilities:');
disp(cellstr([num2str(100 * prediction), repmat('%', numel(prediction), 1)]));

% Display the actual target class
fprintf('Actual Target Class: test_%s\n', trainImgNames(actualClass));

% Display the classification results
displayClassificationResults(predictedClass, trainImgNames);

% Function to preprocess training images
function trainImages = preprocessImages(trainImgNames)
    trainImages = [];
    
    for trainImgName = trainImgNames
        img = imread(trainImgName);
        img = preprocessImage(img);
        trainImages = [trainImages; img];
        
        figure()
        imshow(img);
    end
    
    trainImages = trainImages';
    trainImages = logical(trainImages);
end

% Function to preprocess a single image
function processedImg = preprocessImage(img)
    img = imresize(img, [720 1600], 'bilinear');
    grayThreshold = graythresh(img) * 0.80;
    grayImg = rgb2gray(img);
    bwImg = imbinarize(grayImg, grayThreshold);
    roiImg = imcrop(bwImg, [0 720-250 410 250]);
    roiImg = bwareaopen(roiImg, 30000);
    roiImg = imresize(roiImg, [62 107], 'bilinear');
    processedImg = reshape(roiImg, [1 6634]);
end

% Function to create and train the neural network
function net = createAndTrainNetwork(trainImages, output)
    net = patternnet([6 3], 'trainlm');
    net.trainParam.lr = 0.1;
    net.trainParam.epochs = 40;
    net.trainParam.min_grad = 0;
    net.divideParam.trainRatio = 100/100;
    net.divideParam.valRatio = 0/100;
    net.divideParam.testRatio = 0/100;
    net = train(net, trainImages, output);
end

% Function to display training results
function displayTrainingResults(net, trainImages, output)
    toutput = sim(net, trainImages);
    perfd = perform(net, toutput, output);
    
    disp('Simulated Outputs:');
    disp(toutput');
    
    binaryOutputs = round(toutput);
    
    disp('Binary Outputs (rounded):');
    disp(binaryOutputs');
    
    disp('Actual Target Outputs:');
    disp(output');
    
    correctPredictions = sum(all(binaryOutputs' == output, 1));
    totalSamples = size(output, 2);
    accuracy = correctPredictions / totalSamples * 100;
    
    disp(['Classification Accuracy: ', num2str(accuracy), '%']);
end

% Function to preprocess a test image and get actual class
function [testImg, actualClass] = preprocessTestImage(testImgName)
    img = imread(testImgName);
    img = imresize(img, [720 1600], 'bilinear');
    grayThreshold = graythresh(img) * 0.98;
    grayImg = rgb2gray(img);
    bwImg = imbinarize(grayImg, grayThreshold);
    roiImg = imcrop(bwImg, [0 720-250 410 250]);
    roiImg = bwareaopen(roiImg, 30000);
    roiImg = imresize(roiImg, [62 107], 'bilinear');
    testImg = reshape(roiImg, [1 6634]);
    testImg = testImg';

    % Determine the actual class of the test image
    switch testImgName
        case 'test_5tl.jpg'
            actualClass = 1;
        case 'test_10tl.jpg'
            actualClass = 2;
        case 'test_20tl.jpg'
            actualClass = 3;
        case 'test_50tl.jpg'
            actualClass = 4;
        case 'test_100tl.jpg'
            actualClass = 5;
        case 'test_200tl.jpg'
            actualClass = 6;
        otherwise
            actualClass = -1; % Unknown class
    end
end

% Function to display classification results
function displayClassificationResults(predictedClass, trainImgNames)
    fprintf('Paper bill is classified as %s\n', erase(trainImgNames(predictedClass), '.jpg'));
end
