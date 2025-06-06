# SARIMA Seasonal Time Series Forecasting

This repository contains a project to model and forecast seasonal time series data using the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model. The goal is to capture seasonal patterns and improve forecasting accuracy over traditional ARIMA models.

---

## Project Overview

Seasonal time series data often exhibits repeating patterns at fixed intervals (e.g., monthly, quarterly). SARIMA models are designed to capture both non-seasonal and seasonal components of such data.

In this project, we will:

- Analyze and visualize the time series data.
- Identify seasonal and non-seasonal parameters for the SARIMA model.
- Fit and validate the SARIMA model.
- Forecast future data points and evaluate model performance.

---

## Dataset

The dataset used is publicly available and contains [describe dataset briefly, e.g., monthly airline passengers from 1949 to 1960].

- Source: [NOAA / Kaggle / AirPassengers Dataset / link to dataset]
- Format: CSV file with time series data indexed by date.

---

## Installation

To run this project, you need Python 3.x and the following libraries:

```bash
pip install numpy pandas matplotlib statsmodels pmdarima
