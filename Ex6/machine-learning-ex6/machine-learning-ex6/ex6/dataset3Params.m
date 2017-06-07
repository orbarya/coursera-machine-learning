function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CArr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaArr = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
minCVError = inf;
C = 0.01;
sigma = 0.01;

for i=1 : size(CArr)
    for j=1 : size(sigmaArr)
    
        model = svmTrain (X,y, CArr(i), @(x1, x2) gaussianKernel(x1,x2,sigmaArr(j)), 1e-3, 20);
        predictions = svmPredict(model, Xval);
        CVError = mean(double(predictions ~= yval));
        if (CVError < minCVError)
           minCVError = CVError; 
           C = CArr(i);
           sigma = sigmaArr(j);
        end
    end
end 
% =========================================================================

end
