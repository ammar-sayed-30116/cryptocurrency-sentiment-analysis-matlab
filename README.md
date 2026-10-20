# Cryptocurrency Sentiment Analysis Using MATLAB

## Overview

This project investigates whether cryptocurrency market data and sentiment-related features can help predict future price changes and price direction using MATLAB.

## Objectives

- Prepare and analyse cryptocurrency market and sentiment data.
- Train and compare regression models using market features, sentiment features, and both combined.
- Evaluate directional classification into Down, Unchanged, and Up.
- Examine feature importance and sentiment correlations.
- Evaluate an illustrative sentiment-based trading strategy on the test period.

## Dataset

The dataset contains 18,069 observations and 12 variables, including timestamps, OHLC price data, trading volume, and sentiment statistics.

The prediction target is the price change at the next observed row. Since some timestamps are not exactly one minute apart, the target does not always represent the next minute.

Dataset source and usage permissions: **[Add the original dataset source and licence information here.]**

## Methodology

The data was chronologically divided into training and testing sets, using approximately 70% for training and 30% for testing.

Regression models were evaluated using RMSE, MAE, and directional accuracy. Classification models were evaluated using accuracy, balanced accuracy, macro F1-score, and confusion matrices.

## Key Results

- The combined regression model achieved an RMSE of approximately $5.31, compared with $4.35 for the zero-change baseline.
- Classification accuracy increased from approximately 32.04% for the market-only model to 32.58% for the market-plus-sentiment model.
- The illustrative trading strategy returned approximately -7.30%, compared with +3.22% for buy-and-hold during the tested period.

## Conclusion

The results show that adding sentiment produced only a small improvement in directional classification and did not establish reliable price prediction or a profitable trading strategy.

## How to Run

1. Install MATLAB with the required Statistics and Machine Learning Toolbox functionality.
2. Place the dataset in the expected location.
3. Open `crypto_sentiment_analysis.m`.
4. Run the complete script.
5. Review the generated figures, CSV files, and saved MAT file.

## Limitations

- The results apply to the supplied dataset and tested models.
- Some timestamps have gaps.
- Model performance may vary with data, model settings, and MATLAB version.
- The trading simulation is illustrative and does not establish real-world profitability.

## Disclaimer

This project is for educational purposes only and is not financial advice.
