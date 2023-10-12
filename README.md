# Stock Price Prediction with LSTM

## Overview
This repository contains the implementation of a Long Short-Term Memory (LSTM) model to predict the stock prices of QQQ (Invesco QQQ ETF) at various intervals - daily, 5-minute, 15-minute, and 30-minute.

## Methods and Technologies Used
- **Data Source**: Stock data is fetched using `yfinance`.
- **Modeling**: Implemented LSTM network using `TensorFlow` and `Keras`.
- **Data Visualization**: Leveraged `plotly` for interactive data visualization.
- **Data Preprocessing**: Utilized `pandas` and `NumPy` for data manipulation and `scikit-learn` for scaling.
- **Metrics Calculation**: Used `scikit-learn` to compute model evaluation metrics.

## Model Configuration for Different Intervals
### 1. Daily Interval Data
   - Sequence Length: 240 days [Adjustable]
   - Future Predictions: 120 days [Adjustable]
   
### 2. 5-Minute Interval Data
   - Sequence Length: 60 intervals [Adjustable]
   - Future Predictions: 120 intervals [Adjustable]
   
### 3. 15-Minute Interval Data
   - Sequence Length: 104 intervals [Adjustable]
   - Future Predictions: 26 intervals [Adjustable]
   
### 4. 30-Minute Interval Data
   - Sequence Length: 52 intervals [Adjustable]
   - Future Predictions: 13 intervals [Adjustable]
   
## Metrics
Evaluation metrics include:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

## Visualization
Interactive plots show:
- Actual Prices
- Predicted Prices
- Future Predictions

## Setup & Usage
### Prerequisites
- Python 3.x
- Jupyter Notebook (if running locally)
   
### Required Libraries
```shell
pip install yfinance pandas numpy scikit-learn tensorflow keras plotly
```
## Running the Code

1. **Clone the repository**
2. **Install the required libraries**:
   ```shell
   pip install yfinance pandas numpy scikit-learn tensorflow keras plotly
3. **Run the Jupyter Notebook or Python script**

## Future Work

- Implement different model architectures.
- Optimize the LSTM model for better performance.
- Explore additional features for model enhancement.

