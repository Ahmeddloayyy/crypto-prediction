# crypto-prediction
## Abstract
This project outlines the development and implementation of a cryptocurrency predictive system using advanced machine learning techniques. The system leverages methods such as anomaly detection, clustering, and various forecasting models to predict cryptocurrency prices and generate actionable trading insights. An interactive GUI built with Python facilitates real-time data visualization, empowering traders to make data-driven decisions in a volatile market.

## Table of Contents
Introduction

Aims & Objectives

Methodology

Data Fetching & Preprocessing

Clustering

Correlation Analysis

Exploratory Data Analysis (EDA)

Machine Learning Models

Prophet's Model
ARIMA
Holt-Winters Exponential Smoothing
XGBoost
Support Vector Regression (SVR)

Evaluation

Conclusion

## Introduction
The cryptocurrency market is one of the most dynamic and volatile financial sectors, with a market size exceeding $2 trillion as of 2021. This project, developed as part of the Applied AI in Business course, addresses the challenges of forecasting in such an environment. By analyzing five years of historical price data, the system applies multiple machine learning models to provide traders with reliable predictions and actionable insights.

## Aims & Objectives
Aims:

Develop a predictive model that forecasts future cryptocurrency prices with high accuracy.
Enhance decision-making for cryptocurrency traders by offering actionable trading insights.
Objectives:

Implement and evaluate several machine learning models (ARIMA, Prophet, XGBoost, Exponential Smoothing, and SVR) to determine their effectiveness in predicting cryptocurrency prices.
Utilize five years of historical price data for training and testing.
Develop a user-friendly graphical user interface (GUI) for real-time data visualization and interaction.
Assess the performance of each model using appropriate statistical metrics.

## Methodology

Data Fetching & Preprocessing

Data Collection: Obtained five years of historical cryptocurrency price data.

Cleaning: Missing values were handled via forward-fill and backward-fill, and anomalies were filtered to improve data quality.

Transformation: The data was transposed so that each row represents a cryptocurrency ticker and columns represent prices over time.

Clustering
Dimensionality Reduction: Principal Component Analysis (PCA) was used to reduce the dataset’s dimensions while retaining significant variance.

Clustering: K-Means clustering partitioned the data into four clusters, grouping similar cryptocurrencies. The results were visualized using scatter plots of the two principal components.

Correlation Analysis

Correlation Matrix: Computed for selected tickers (e.g., SOL-USD, BTC-USD, ETH-USD, XMR-USD) to understand their linear relationships.

Visualization: A heatmap was generated to highlight the strongest positive and negative correlations among cryptocurrencies.

## Exploratory Data Analysis (EDA)

### Analyses Performed:

Historical price trends

Price distribution analysis

Daily percentage returns

Distribution of daily returns

Purpose: These analyses identified patterns, outliers, and trends, guiding subsequent model development.

## Machine Learning Models
### Prophet's Model
Prepared the data to match Prophet’s expected format (ds for date, y for price).
Configured the model with yearly seasonality to capture long-term trends.
Forecasted 30 days beyond the available data and visualized the predictions.
### ARIMA
Split the time series into training and testing sets.
Used auto ARIMA to select optimal parameters via the Akaike Information Criterion (AIC).
Forecasted the next 30 days and compared them to actual data.
### Holt-Winters Exponential Smoothing
Employed a multiplicative trend without seasonality.
Trained on all but the last 30 days, then forecasted and compared to actual prices.
### XGBoost
Utilized a sliding window approach to create sequential features from historical prices.
Configured the model with 100 estimators and a learning rate of 0.1.
Evaluated performance on an 80:20 train-test split.
### Support Vector Regression (SVR)
Used a sliding window size of 5 for feature generation.
Normalized features with MinMaxScaler.
Configured SVR with an RBF kernel, C=1000, and gamma=0.1, and assessed forecasting accuracy.
## Evaluation
Each model was evaluated using statistical metrics (e.g., accuracy, mean squared error, etc.). Key findings:

### ARIMA & Prophet: Strong at capturing seasonal and non-linear trends.
### Holt-Winters: Effective for data with gradual trends.
### XGBoost & SVR: Excelled in capturing non-linear patterns with a sliding window approach.
# Conclusion
The cryptocurrency predictive system demonstrates how machine learning can tackle the challenges of a volatile market. By integrating anomaly detection, clustering, EDA, and multiple forecasting models, the system provides traders with robust insights. Future enhancements could include real-time data integration, ensemble methods, and more advanced hyperparameter tuning to further improve accuracy.
