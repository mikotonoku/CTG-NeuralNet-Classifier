% Loading data from the file
data = load('CTGdata.mat');
targets = dummyvar(data.typ_ochorenia);                                     % Creating a binary representation of categorical values

% Creating the neural network structure
hidden_neurons = 10;                                                        % Initial number of neurons in the hidden layer
net = patternnet(hidden_neurons);                                           % Creating the network

% Neural network structure
% Setting parameters for splitting data into training and testing sets
net.divideFcn = 'dividerand'; 
net.divideParam.trainRatio = 0.6;  
net.divideParam.valRatio = 0; 
net.divideParam.testRatio = 0.4;

net.trainParam.goal = 0.001;                                                % Target error value that the network aims to achieve during training
net.trainParam.epochs = 1000;                                               % Maximum number of epochs
net.trainParam.max_fail = 12;                                               % Maximum number of validation failures (if validation is enabled)

% Training the network
[net, tr] = train(net, data.NDATA', targets');

view(net)

% Extracting training and testing data based on indices from `tr`
train_data = data.NDATA(tr.trainInd, :)';
train_target = targets(tr.trainInd, :)';

test_data = data.NDATA(tr.testInd, :)';
test_target = targets(tr.testInd, :)';

% Calculating network outputs for training data
train_outputs = net(train_data);
train_accuracies = 1 - confusion(train_target, train_outputs);              % Accuracy values
train_accuracies = train_accuracies * 100;                                  % Percentage error

% Calculating network outputs for test data
test_outputs = net(test_data);
test_accuracies = 1 - confusion(test_target, test_outputs);                 % Accuracy values
test_accuracies = test_accuracies * 100;
testError = 100 - test_accuracies;                                          % Percentage error

fprintf('Training Accuracy: min = %.2f%%, avg = %.2f%%, max = %.2f%%\n', ...
    min(train_accuracies), mean(train_accuracies), max(train_accuracies));
fprintf('Testing Accuracy: min = %.2f%%, avg = %.2f%%, max = %.2f%%\n', ...
    min(test_accuracies), mean(test_accuracies), max(test_accuracies));

% Visualizing the confusion matrix for training data
figure;
plotconfusion(train_target, train_outputs);
title('Training Data');

% Visualizing the confusion matrix for test data
figure;
plotconfusion(test_target, test_outputs);
title(sprintf('Test Data\nError: %.2f%%', testError));

% Calculating network outputs for overall data (optional)
overall_target = [train_target, test_target];
overall_result = [train_outputs, test_outputs];

figure;
plotconfusion(overall_target, overall_result);
title('Overall Data');

%--------------------------------------------------------------------------

% Selecting one sample from each disease type
sample_normal = data.NDATA(find(data.typ_ochorenia == 1, 1), :)';
sample_suspect = data.NDATA(find(data.typ_ochorenia == 2, 1), :)';
sample_pathological = data.NDATA(find(data.typ_ochorenia == 3, 1), :)';

% Computing outputs for each sample
output_normal = net(sample_normal);
output_suspect = net(sample_suspect);
output_pathological = net(sample_pathological);

% Assigning categories
[~, group_normal] = max(output_normal);
[~, group_suspect] = max(output_suspect);
[~, group_pathological] = max(output_pathological);

% Displaying results
fprintf('Normal Sample: Predicted probabilities = [%s], Group = %d\n', ...
    num2str(output_normal'), group_normal);
fprintf('Suspect Sample: Predicted probabilities = [%s], Group = %d\n', ...
    num2str(output_suspect'), group_suspect);
fprintf('Pathological Sample: Predicted probabilities = [%s], Group = %d\n', ...
    num2str(output_pathological'), group_pathological);

%--------------------------------------------------------------------------
% Computing confusion matrix and accuracy for training data
[~, cm_train, ~, train_per] = confusion(train_target, train_outputs);
train_per = train_per';                                                     % Transposing for better readability

% Computing confusion matrix and accuracy for test data
[~, cm_test, ~, test_per] = confusion(test_target, test_outputs);
test_per = test_per';                                                       % Transposing for better readability

% Sensitivity (true positive rate) - Groups 2 and 3 are considered positive
sensitivity_train = sum(cm_train(2:3, 2:3), 'all') / sum(cm_train(2:3, :), 'all');
sensitivity_test = sum(cm_test(2:3, 2:3), 'all') / sum(cm_test(2:3, :), 'all');

% Specificity (true negative rate) - Group 1 is considered negative
specificity_train = cm_train(1, 1) / sum(cm_train(1, :));
specificity_test = cm_test(1, 1) / sum(cm_test(1, :));

% Accuracy (overall classification success)
accuracy_train = sum(diag(cm_train)) / sum(cm_train(:));
accuracy_test = sum(diag(cm_test)) / sum(cm_test(:));

% Displaying results for the training set
fprintf('Training Set:\n');
fprintf('Sensitivity: %.2f%%\n', sensitivity_train * 100);
fprintf('Specificity: %.2f%%\n', specificity_train * 100);
fprintf('Accuracy: %.2f%%\n', accuracy_train * 100);

% Displaying results for the test set
fprintf('\nTest Set:\n');
fprintf('Sensitivity: %.2f%%\n', sensitivity_test * 100);
fprintf('Specificity: %.2f%%\n', specificity_test * 100);
fprintf('Accuracy: %.2f%%\n', accuracy_test * 100);
