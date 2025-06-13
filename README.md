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

The dataset used is publicly available and contains [describe dataset briefly, e.g., monthly airline passengers from **2005-01-01** to **2024-12-31**].

- Source: [NOAA]
- Format: API JSON
- Columns: `date`, `AverageTemperature`
```
- Date Range: **2005-01-01** to **2024-12-31**
```
- Frequency: Monthly
```
- Data Source: [NOAA API](https://www.ncdc.noaa.gov/cdo-web/webservices/v2)
```
- Data Format: JSON
```
- Data Columns: `date`, `AverageTemperature`
```

---

## Installation

To run this project, you need Python 3.x and the following libraries:

```bash
pip install -r requirements.txt

