load("ad_data.mat")
load("feature_name.mat")

% Run l1 logistic
[weights, c_val] = l1_train(X_train, y_train, 1e-8);

% Count number of non-zero weights
num_non_zero = length(weights(weights ~= 0));

% Get predictions (probs) based on logistic function
raw_preds = X_test * weights + c_val;
preds = 1./ (1.+ exp(-raw_preds));

% Calculate the AUC score
[X,Y,T,AUC] = perfcurve(y_test, preds, '1');


% Function for training using L1 Logistic regression
function [w, c] = l1_train(data, labels, par)
    % OUTPUT w is equivalent to the first d dimension of weights in logistic train
    % c is the bias term, equivalent to the last dimension in weights in logistic train.
   
    % Specify the options (use without modification).
    opts.rFlag = 1; % range of par within [0, 1].
    opts.tol = 1e-6; % optimization precision
    opts.tFlag = 4; % termination options.
    opts.maxIter = 5000; % maximum iterations.

    [w, c] = LogisticR(data, labels, par, opts);
end



