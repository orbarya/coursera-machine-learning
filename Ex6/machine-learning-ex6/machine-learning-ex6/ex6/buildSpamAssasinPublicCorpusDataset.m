clear ; close all; clc

%x=[];
%y=[];
% %   1. 20021010_easy_ham.tar\20021010_easy_ham\easy_ham
% %   Non spam
% dirRelNames = cell(9,2);
% dirRelNames{1,1} = '.\spam_assasin_mail_corpus\20021010_easy_ham.tar\20021010_easy_ham\easy_ham\';
% dirRelNames{1,2} = 0;
% dirRelNames{2,1} = '.\spam_assasin_mail_corpus\20021010_hard_ham.tar\20021010_hard_ham\hard_ham\';
% dirRelNames{2,2} = 0;
% dirRelNames{3,1} = '.\spam_assasin_mail_corpus\20021010_spam.tar\20021010_spam\spam\';
% dirRelNames{3,2} = 1;
% dirRelNames{4,1} = '.\spam_assasin_mail_corpus\20030228_easy_ham.tar\20030228_easy_ham\easy_ham\';
% dirRelNames{4,2} = 0;
% dirRelNames{5,1} = '.\spam_assasin_mail_corpus\20030228_easy_ham_2.tar\20030228_easy_ham_2\easy_ham_2\';
% dirRelNames{5,2} = 0;
% dirRelNames{6,1} = '.\spam_assasin_mail_corpus\20030228_hard_ham.tar\20030228_hard_ham\hard_ham\';
% dirRelNames{6,2} = 0;
% dirRelNames{7,1} = '.\spam_assasin_mail_corpus\20030228_spam.tar\20030228_spam\spam\';
% dirRelNames{7,2} = 1;
% dirRelNames{8,1} = '.\spam_assasin_mail_corpus\20030228_spam_2.tar\20030228_spam_2\spam_2\';
% dirRelNames{8,2} = 1;
% dirRelNames{9,1} = '.\spam_assasin_mail_corpus\20050311_spam_2.tar\20050311_spam_2\spam_2\';
% dirRelNames{9,2} = 1;
%             
% 
% for i=1:size(dirRelNames)
%     files = dir(strcat (dirRelNames{i,1}, '*'));
%     filesNum = 0;
%     for file = files'
%         if (file.name =='.')
%             continue;
%         end
%         fileContent = fileread(strcat(dirRelNames{i,1}, file.name));
% 
%         word_indices = processEmail (fileContent);
%         features = emailFeatures (word_indices);
%         x = [x ; features];    
%         filesNum = filesNum + 1;
%     end   
%     y = [y ;ones(filesNum,1) * dirRelNames{i,2}];
% end

% load ('.\spam_assasin_mail_corpus\SpamAssasinCorpusDataset.mat');
% % Divide the dataset to training set, cross validation set and test set.
% samplesNumber = size(x,1);
% samplesIndices = 1:samplesNumber;
% TRAINING_SET_PERCENTAGE = 0.6;
% trainingSamplesNum = round(samplesNumber * TRAINING_SET_PERCENTAGE);
% 
% % Indices of training set in x.
% training_indices = randperm (samplesNumber, trainingSamplesNum);
% 
% % Indices in x of samples not in training set.
% samplesIndicesNoTraining = samplesIndices(~ismember (samplesIndices, training_indices));
% CV_PERCENTAGE = 0.2;
% cvSamplesNum = round(samplesNumber * CV_PERCENTAGE);
% samplesNoTrainingNumber = size(samplesIndicesNoTraining,2);
% 
% % cv_indices_temp holds Indices of cv in samplesIndicesNoTraining.
% % In other words, given a number, a, in cv_indices_temp,
% % samplesIndicesNoTraining(a) holds an index of a cv sample in x.
% cv_indices_temp = randperm (samplesNoTrainingNumber, cvSamplesNum);
% 
% % cv_indices holds The cv indices in x.
% cv_indices = samplesIndicesNoTraining (cv_indices_temp);
% 
% % Now we want to find the complementary indices in samplesIndicesNoTraining
% % to cv_indices_temp.
% all_indices = 1:samplesNoTrainingNumber;
% test_indices = samplesIndicesNoTraining(~ismember (all_indices, cv_indices_temp));
% 
% Xtraining = x(training_indices, :);
% YTraining = y(training_indices, :);
% 
% xCV = x(cv_indices, :);
% yCV = y(cv_indices, :);
% 
% xTest = x(test_indices, :);
% yTest = y(test_indices, :);




%% =========== Part 3: Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.

% Load the Spam Email dataset
% You will have X, y in your environment
load('./spam_assasin_mail_corpus/SpamAssasinCorpusDataset.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(Xtraining, YTraining, C, @linearKernel);

p = svmPredict(model, Xtraining);

fprintf('Training Accuracy: %f\n', mean(double(p == YTraining)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, xTest);

fprintf('Test Accuracy: %f\n', mean(double(p == yTest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
%

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end