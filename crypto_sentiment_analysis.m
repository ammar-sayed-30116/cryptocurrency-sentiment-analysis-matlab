
% CRYPTOCURRENCY SENTIMENT ANALYSIS AND TRADING
% Project: Sentiment Analysis in Cryptocurrency Trading
%
% Models:
% 1. Zero-change regression baseline
% 2. Market-only regression
% 3. Sentiment-only regression
% 4. Market + sentiment regression
% 5. Majority-class classification baseline
% 6. Market-only classification
% 7. Market + sentiment classification
%
% IMPORTANT:
% Predictions use information from observation t to predict
% the change from Close(t) to Close(t+1).
% Market features must be available before the prediction is made.

clc;
clear;
close all;

% Reproducible randomized model training
rng(42, 'twister');

fprintf('\n========================================\n');
fprintf(' CRYPTOCURRENCY SENTIMENT ANALYSIS\n');
fprintf('========================================\n\n');

%% 1. LOAD AND VALIDATE DATA

filename = 'filtered_merged_sentiment_price1.csv';

if ~isfile(filename)
    error('Dataset not found: %s. Place it in the Current Folder.', ...
        filename);
end

T = readtable(filename);

requiredColumns = {
    'Datetime'
    'GroupCount'
    'mean_Sentiment'
    'median_Sentiment'
    'max_Sentiment'
    'min_Sentiment'
    'var_Sentiment'
    'Open'
    'High'
    'Low'
    'Close'
    'Volume'
};

missingColumns = setdiff(requiredColumns, ...
    T.Properties.VariableNames);

if ~isempty(missingColumns)
    error('Missing required columns: %s', ...
        strjoin(missingColumns, ', '));
end

if ~isdatetime(T.Datetime)
    T.Datetime = datetime(T.Datetime);
end

if any(isnat(T.Datetime))
    error('Datetime contains missing or invalid timestamps.');
end

% Sort chronologically
T = sortrows(T, 'Datetime');

% Validate numeric columns
numericColumns = requiredColumns(2:end);

for k = 1:numel(numericColumns)
    col = numericColumns{k};

    if ~isnumeric(T.(col))
        error('Column %s must contain numeric data.', col);
    end

    T.(col) = double(T.(col));
end

% Reject duplicate or non-increasing timestamps.
% Duplicates should be resolved according to the dataset's
% meaning rather than silently removed.
timeGaps = diff(T.Datetime);

if any(timeGaps <= minutes(0))
    error(['Duplicate or non-increasing timestamps detected. ' ...
        'Resolve them before modeling.']);
end

fprintf('Dataset loaded successfully.\n');
fprintf('Total observations: %d\n', height(T));
fprintf('Total variables: %d\n', width(T));

%% 2. TIMESTAMP CHECK

fprintf('\n===== TIMESTAMP CHECK =====\n');

if isempty(timeGaps)
    oneMinutePercent = NaN;
else
    oneMinutePercent = mean(timeGaps == minutes(1)) * 100;
end

fprintf('Exactly one-minute intervals: %.2f%%\n', ...
    oneMinutePercent);

if ~isempty(timeGaps)
    fprintf('Median time gap: %.2f minutes\n', ...
        median(minutes(timeGaps)));
end

if any(timeGaps ~= minutes(1))
    fprintf(['WARNING: Some intervals are not one minute.\n' ...
        'The target predicts the next observed row, not ' ...
        'necessarily the next minute.\n']);
end

%% 3. CREATE FEATURES AND TARGET

featureNames = {
    'GroupCount'
    'mean_Sentiment'
    'median_Sentiment'
    'max_Sentiment'
    'min_Sentiment'
    'var_Sentiment'
    'Open'
    'High'
    'Low'
    'Close'
    'Volume'
};

% Features at observation t
XAll = T{1:end-1, featureNames};

% Target: next observed closing price minus current close
yAll = T.Close(2:end) - T.Close(1:end-1);

% Target row identifies the row containing the future close
targetRowsAll = (2:height(T))';

% Keep only rows with valid features and targets
valid = all(isfinite(XAll), 2) & isfinite(yAll);

X = XAll(valid, :);
y = yAll(valid);
targetRows = targetRowsAll(valid);

if numel(y) < 10
    error('Too few valid observations to train the models.');
end

fprintf('\nValid observations: %d\n', numel(y));

%% 4. CHRONOLOGICAL TRAIN/TEST SPLIT

% First 70%% for training, last 30%% for testing.
% Do not randomly shuffle time-series observations.

n = numel(y);
nTrain = floor(0.70 * n);

if nTrain < 2 || nTrain >= n
    error('Insufficient observations for the selected split.');
end

XTrain = X(1:nTrain, :);
yTrain = y(1:nTrain);

XTest = X(nTrain+1:end, :);
yTest = y(nTrain+1:end);

trainRows = targetRows(1:nTrain);
testRows = targetRows(nTrain+1:end);

fprintf('Training observations: %d\n', numel(yTrain));
fprintf('Testing observations: %d\n', numel(yTest));

%% 5. DEFINE MARKET AND SENTIMENT FEATURES

marketFeatures = {'Open','High','Low','Close','Volume'};

sentimentFeatures = {
    'GroupCount'
    'mean_Sentiment'
    'median_Sentiment'
    'max_Sentiment'
    'min_Sentiment'
    'var_Sentiment'
};

marketIndices = find(ismember(featureNames, marketFeatures));
sentimentIndices = find(ismember(featureNames, sentimentFeatures));

%% 6. TRAIN FOUR REGRESSION MODELS

fprintf('\n===== TRAINING REGRESSION MODELS =====\n');

% A. Zero-change baseline
predZero = zeros(size(yTest));

% B. Market-only model
modelMarket = fitrensemble( ...
    XTrain(:,marketIndices), yTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100);

predMarket = predict( ...
    modelMarket, XTest(:,marketIndices));

% C. Sentiment-only model
modelSentiment = fitrensemble( ...
    XTrain(:,sentimentIndices), yTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100);

predSentiment = predict( ...
    modelSentiment, XTest(:,sentimentIndices));

% D. Market + sentiment model
modelFull = fitrensemble( ...
    XTrain, yTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100);

predFull = predict(modelFull, XTest);

fprintf('All four regression models trained.\n');

%% 7. REGRESSION MODEL COMPARISON

regressionPredictions = [ ...
    predZero, predMarket, predSentiment, predFull];

regressionNames = [
    "Zero-change baseline"
    "Market only"
    "Sentiment only"
    "Market + sentiment"
];

numberModels = size(regressionPredictions, 2);

rmseValues = zeros(numberModels,1);
maeValues = zeros(numberModels,1);
directionValues = zeros(numberModels,1);

nonzeroActual = yTest ~= 0;

for k = 1:numberModels

    pred = regressionPredictions(:,k);

    rmseValues(k) = sqrt(mean((yTest - pred).^2));
    maeValues(k) = mean(abs(yTest - pred));

    if any(nonzeroActual)
        directionValues(k) = 100 * mean( ...
            sign(pred(nonzeroActual)) == ...
            sign(yTest(nonzeroActual)));
    else
        directionValues(k) = NaN;
    end
end

ModelComparison = table( ...
    regressionNames, rmseValues, maeValues, directionValues, ...
    'VariableNames', { ...
    'Model','TestRMSE_USD','TestMAE_USD', ...
    'DirectionalAccuracyPercent'});

fprintf('\n===== REGRESSION MODEL COMPARISON =====\n');
disp(ModelComparison);

% Full-model metrics
predictedChange = predFull;

rmse = rmseValues(4);
mae = maeValues(4);
baselineRMSE = rmseValues(1);
directionalAccuracy = directionValues(4);

ssResidual = sum((yTest - predictedChange).^2);
ssTotal = sum((yTest - mean(yTest)).^2);

if ssTotal > 0
    rSquared = 1 - ssResidual / ssTotal;
else
    rSquared = NaN;
end

fprintf('\n===== FULL REGRESSION MODEL =====\n');
fprintf('RMSE: %.4f USD\n', rmse);
fprintf('MAE: %.4f USD\n', mae);
fprintf('R-squared: %.4f\n', rSquared);
fprintf('Zero-change baseline RMSE: %.4f USD\n', baselineRMSE);
fprintf('Directional accuracy: %.2f%%\n', directionalAccuracy);

if rmse < baselineRMSE
    fprintf('Full model beats the zero-change baseline on RMSE.\n');
else
    fprintf('Full model does not beat the zero-change baseline on RMSE.\n');
end

if rmseValues(3) < rmseValues(2)
    fprintf('Sentiment-only RMSE is lower than market-only RMSE.\n');
else
    fprintf('Sentiment-only RMSE is not lower than market-only RMSE.\n');
end

if rmseValues(4) < rmseValues(2)
    fprintf('Combined model RMSE improves on market-only RMSE.\n');
else
    fprintf('Combined model RMSE does not improve on market-only RMSE.\n');
end

%% 8. RECONSTRUCT TEST-PERIOD PRICES

% testRows identifies the target (future-close) row.
% The preceding close is the price at the feature observation.

previousClose = T.Close(testRows - 1);
actualPrice = T.Close(testRows);
predictedPrice = previousClose + predictedChange;

PredictionResults = table( ...
    T.Datetime(testRows), ...
    yTest, ...
    predictedChange, ...
    actualPrice, ...
    predictedPrice, ...
    previousClose, ...
    'VariableNames', { ...
    'Datetime','ActualPriceChange','PredictedPriceChange', ...
    'ActualClose','PredictedClose','PreviousCloseReference'});

%% 9. FEATURE IMPORTANCE

importance = predictorImportance(modelFull);

ImportanceTable = table( ...
    string(featureNames(:)), importance(:), ...
    'VariableNames', {'Feature','Importance'});

ImportanceTable = sortrows( ...
    ImportanceTable, 'Importance', 'descend');

fprintf('\n===== FEATURE IMPORTANCE =====\n');
disp(ImportanceTable);

%% 10. SENTIMENT CORRELATION ANALYSIS

sentimentX = X(:,sentimentIndices);

sentimentCorrelations = zeros(numel(sentimentIndices),1);

for k = 1:numel(sentimentIndices)
    R = corrcoef(sentimentX(:,k), y);
    sentimentCorrelations(k) = R(1,2);
end

SentimentCorrelationTable = table( ...
    string(sentimentFeatures(:)), sentimentCorrelations, ...
    'VariableNames', {'SentimentFeature','CorrelationWithNextChange'});

SentimentCorrelationTable = sortrows( ...
    SentimentCorrelationTable, ...
    'CorrelationWithNextChange', 'descend');

fprintf('\n===== SENTIMENT CORRELATIONS =====\n');
disp(SentimentCorrelationTable);

% Correlation is association, not proof of causation or predictability.

%% 11. DIRECTIONAL CLASSIFICATION

fprintf('\n===== DIRECTIONAL CLASSIFICATION =====\n');

% 1 = Down, 2 = Unchanged, 3 = Up
classLabels = ["Down"; "Unchanged"; "Up"];

classY = ones(size(y));
classY(y == 0) = 2;
classY(y > 0) = 3;

classTrain = classY(1:nTrain);
classTest = classY(nTrain+1:end);

% Baseline predicts the most common class in training data
trainClassCounts = accumarray(classTrain, 1, [3 1]);
[~, majorityClass] = max(trainClassCounts);

predClassBaseline = repmat( ...
    majorityClass, size(classTest));

% Market-only classifier
classMarketModel = fitcensemble( ...
    XTrain(:,marketIndices), classTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100);

predClassMarket = predict( ...
    classMarketModel, XTest(:,marketIndices));

% Market + sentiment classifier
classFullModel = fitcensemble( ...
    XTrain, classTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100);

predClassFull = predict(classFullModel, XTest);

% Confusion matrices: rows = actual, columns = predicted
Cbaseline = confusionmat( ...
    classTest, predClassBaseline, 'Order', 1:3);

Cmarket = confusionmat( ...
    classTest, predClassMarket, 'Order', 1:3);

Cfull = confusionmat( ...
    classTest, predClassFull, 'Order', 1:3);

fprintf('\nTraining class distribution:\n');

for k = 1:3
    fprintf('%s: %d (%.2f%%)\n', ...
        classLabels(k), trainClassCounts(k), ...
        100 * trainClassCounts(k) / numel(classTrain));
end

%% 12. CLASSIFICATION METRICS

classificationPredictions = { ...
    predClassBaseline, predClassMarket, predClassFull};

classificationConfusions = {Cbaseline, Cmarket, Cfull};

classificationNames = [
    "Majority baseline"
    "Market only"
    "Market + sentiment"
];

nClassModels = numel(classificationPredictions);

accuracy = zeros(nClassModels,1);
balancedAccuracy = zeros(nClassModels,1);
macroF1 = zeros(nClassModels,1);

for m = 1:nClassModels

    pred = classificationPredictions{m};
    C = classificationConfusions{m};

    accuracy(m) = mean(pred == classTest) * 100;

    recall = zeros(3,1);
    precision = zeros(3,1);
    f1 = zeros(3,1);

    for c = 1:3

        actualCount = sum(C(c,:));
        predictedCount = sum(C(:,c));

        if actualCount > 0
            recall(c) = C(c,c) / actualCount;
        end

        if predictedCount > 0
            precision(c) = C(c,c) / predictedCount;
        end

        if precision(c) + recall(c) > 0
            f1(c) = 2 * precision(c) * recall(c) / ...
                (precision(c) + recall(c));
        end
    end

    balancedAccuracy(m) = mean(recall) * 100;
    macroF1(m) = mean(f1) * 100;
end

ClassificationComparison = table( ...
    classificationNames, accuracy, balancedAccuracy, macroF1, ...
    'VariableNames', { ...
    'Model','AccuracyPercent', ...
    'BalancedAccuracyPercent','MacroF1Percent'});

fprintf('\n===== CLASSIFICATION MODEL COMPARISON =====\n');
disp(ClassificationComparison);

% Per-class metrics for the combined classifier
precisionFull = zeros(3,1);
recallFull = zeros(3,1);
f1Full = zeros(3,1);

for c = 1:3

    if sum(Cfull(:,c)) > 0
        precisionFull(c) = Cfull(c,c) / sum(Cfull(:,c));
    end

    if sum(Cfull(c,:)) > 0
        recallFull(c) = Cfull(c,c) / sum(Cfull(c,:));
    end

    if precisionFull(c) + recallFull(c) > 0
        f1Full(c) = 2 * precisionFull(c) * recallFull(c) / ...
            (precisionFull(c) + recallFull(c));
    end
end

ClassificationMetrics = table( ...
    classLabels, ...
    precisionFull * 100, ...
    recallFull * 100, ...
    f1Full * 100, ...
    'VariableNames', { ...
    'Class','PrecisionPercent','RecallPercent','F1Percent'});

fprintf('\n===== COMBINED MODEL: PER-CLASS METRICS =====\n');
disp(ClassificationMetrics);

fprintf('\n===== MARKET-ONLY CONFUSION MATRIX =====\n');
disp(array2table(Cmarket, ...
    'VariableNames', {'PredDown','PredUnchanged','PredUp'}, ...
    'RowNames', {'ActualDown','ActualUnchanged','ActualUp'}));

fprintf('\n===== MARKET + SENTIMENT CONFUSION MATRIX =====\n');
disp(array2table(Cfull, ...
    'VariableNames', {'PredDown','PredUnchanged','PredUp'}, ...
    'RowNames', {'ActualDown','ActualUnchanged','ActualUp'}));

%% 13. ILLUSTRATIVE SENTIMENT BACKTEST

% This is a simplified demonstration, not a validated trading system.
% The sentiment signal at time t is applied to the return from t to t+1.
% Positions are evaluated on the same held-out observations as regression.

sourceRows = testRows - 1;

% Causal rolling mean: current and preceding 14 observations
smoothedSentiment = movmean(T.mean_Sentiment, [14 0]);

testSentiment = smoothedSentiment(sourceRows);

marketReturns = ...
    T.Close(testRows) ./ T.Close(sourceRows) - 1;

threshold = 0.05;
costRate = 0.0005;

position = zeros(size(testSentiment));
position(testSentiment > threshold) = 1;
position(testSentiment < -threshold) = -1;

previousPosition = [0; position(1:end-1)];

% A full reversal from -1 to +1 incurs twice the unit cost
transactionCosts = costRate .* abs(position - previousPosition);

strategyReturns = ...
    position .* marketReturns - transactionCosts;

strategyEquity = cumprod(1 + strategyReturns);
buyHoldEquity = cumprod(1 + marketReturns);

totalStrategyReturn = strategyEquity(end) - 1;
totalBuyHoldReturn = buyHoldEquity(end) - 1;

equityWithInitial = [1; strategyEquity(:)];
peakEquity = cummax(equityWithInitial);
drawdown = equityWithInitial ./ peakEquity - 1;
maxDrawdown = min(drawdown);

numPositionChanges = sum(abs(position - previousPosition) > 0);

fprintf('\n===== ILLUSTRATIVE TEST-PERIOD BACKTEST =====\n');
fprintf('Position changes: %d\n', numPositionChanges);
fprintf('Strategy return: %.2f%%\n', totalStrategyReturn * 100);
fprintf('Buy-and-hold return: %.2f%%\n', totalBuyHoldReturn * 100);
fprintf('Maximum drawdown: %.2f%%\n', maxDrawdown * 100);
fprintf('Transaction cost per unit position change: %.3f%%\n', ...
    costRate * 100);

%% 14. VISUALIZATIONS

% Figure 1: Actual vs predicted price
fig1 = figure('Name','Actual vs Predicted Price');

plot(T.Datetime(testRows), actualPrice, 'LineWidth', 1.1);
hold on;
plot(T.Datetime(testRows), predictedPrice, 'LineWidth', 1.1);
plot(T.Datetime(testRows), previousClose, '--', 'LineWidth', 1);

title('Actual vs Predicted Closing Price');
xlabel('Date and Time');
ylabel('Price (USD)');

legend('Actual Close','Predicted Close', ...
    'Zero-Change (Previous Close)', 'Location','best');

grid on;
hold off;

exportgraphics(fig1, 'actual_vs_predicted_price.png', ...
    'Resolution',300);

% Figure 2: Regression metrics
fig2 = figure('Name','Regression Model Comparison');

bar([rmseValues, maeValues]);

xticks(1:numberModels);
xticklabels(regressionNames);
xtickangle(20);

ylabel('Error (USD)');
title('Regression Model Comparison');
legend('RMSE','MAE','Location','best');
grid on;

exportgraphics(fig2, 'regression_model_comparison.png', ...
    'Resolution',300);

% Figure 3: Feature importance
fig3 = figure('Name','Feature Importance');

barh(importance);
yticks(1:numel(featureNames));
yticklabels(featureNames);
set(gca,'YDir','reverse');

xlabel('Predictor Importance');
title('Full Regression Model: Feature Importance');
grid on;

exportgraphics(fig3, 'feature_importance.png', ...
    'Resolution',300);

% Figure 4: Sentiment correlations
fig4 = figure('Name','Sentiment Correlations');

bar(sentimentCorrelations);
xticks(1:numel(sentimentFeatures));
xticklabels(sentimentFeatures);
xtickangle(25);

ylabel('Correlation with Next Price Change');
title('Sentiment Features vs Next Observed Price Change');
yline(0,'--');
grid on;

exportgraphics(fig4, 'sentiment_correlations.png', ...
    'Resolution',300);

% Figure 5: Backtest equity curves
fig5 = figure('Name','Trading Backtest');

plot(T.Datetime(testRows), strategyEquity, 'LineWidth',1.1);
hold on;
plot(T.Datetime(testRows), buyHoldEquity, '--', 'LineWidth',1.1);

title('Illustrative Sentiment Strategy vs Buy and Hold');
xlabel('Date and Time');
ylabel('Growth of Initial $1');
legend('Sentiment Strategy','Buy and Hold','Location','best');

grid on;
hold off;

exportgraphics(fig5, 'trading_backtest.png', ...
    'Resolution',300);

% Figure 6: Classification comparison
fig6 = figure('Name','Classification Model Comparison');

bar([accuracy, balancedAccuracy, macroF1]);

xticks(1:nClassModels);
xticklabels(classificationNames);
xtickangle(20);

ylabel('Score (%)');
title('Classification Model Performance');
legend('Accuracy','Balanced Accuracy','Macro F1', ...
    'Location','best');

grid on;

exportgraphics(fig6, 'classification_model_comparison.png', ...
    'Resolution',300);

% Figure 7: Market-only confusion matrix
fig7 = figure('Name','Market-Only Confusion Matrix');

confusionchart(Cmarket, {'Down','Unchanged','Up'});
title('Market-Only Classifier');

exportgraphics(fig7, 'confusion_matrix_market_only.png', ...
    'Resolution',300);

% Figure 8: Market + sentiment confusion matrix
fig8 = figure('Name','Market + Sentiment Confusion Matrix');

confusionchart(Cfull, {'Down','Unchanged','Up'});
title('Market + Sentiment Classifier');

exportgraphics(fig8, 'confusion_matrix_market_sentiment.png', ...
    'Resolution',300);

%% 15. SAVE RESULTS

writetable(ModelComparison, ...
    'crypto_model_comparison.csv');

writetable(PredictionResults, ...
    'crypto_prediction_results.csv');

writetable(ImportanceTable, ...
    'crypto_feature_importance.csv');

writetable(SentimentCorrelationTable, ...
    'crypto_sentiment_correlations.csv');

writetable(ClassificationComparison, ...
    'crypto_classification_comparison.csv');

writetable(ClassificationMetrics, ...
    'crypto_classification_metrics.csv');

save('crypto_sentiment_results.mat', ...
    'modelMarket','modelSentiment','modelFull', ...
    'classMarketModel','classFullModel', ...
    'ModelComparison','PredictionResults', ...
    'ImportanceTable','SentimentCorrelationTable', ...
    'ClassificationComparison','ClassificationMetrics', ...
    'Cbaseline','Cmarket','Cfull', ...
    'rmse','mae','rSquared','baselineRMSE', ...
    'directionalAccuracy','totalStrategyReturn', ...
    'totalBuyHoldReturn','maxDrawdown','threshold','costRate');

fprintf('\n========================================\n');
fprintf(' PROJECT EXECUTION COMPLETED\n');
fprintf('========================================\n');

fprintf('Saved CSV files:\n');
fprintf('- crypto_model_comparison.csv\n');
fprintf('- crypto_prediction_results.csv\n');
fprintf('- crypto_feature_importance.csv\n');
fprintf('- crypto_sentiment_correlations.csv\n');
fprintf('- crypto_classification_comparison.csv\n');
fprintf('- crypto_classification_metrics.csv\n');

fprintf('\nSaved model file:\n');
fprintf('- crypto_sentiment_results.mat\n');       

fprintf('\nGenerated figures (8):\n');
fprintf('- actual_vs_predicted_price.png\n');
fprintf('- regression_model_comparison.png\n');
fprintf('- feature_importance.png\n');
fprintf('- sentiment_correlations.png\n');
fprintf('- trading_backtest.png\n');
fprintf('- classification_model_comparison.png\n');
fprintf('- confusion_matrix_market_only.png\n');
fprintf('- confusion_matrix_market_sentiment.png\n');

fprintf('\nInterpret model results cautiously.\n');
fprintf('This project does not establish a profitable trading system.\n');
fprintf('========================================\n');
