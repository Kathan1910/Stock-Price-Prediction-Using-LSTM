#This is a daily script DO NOT chang values to intraday It will not work for that. 

import dash
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
import datetime


# API Key and Fetch Data
API_KEY = 'F6U8EWWSXP0JBP**'
SYMBOL = 'QQQ'
TIME_SERIES_SIZE = 'full'
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&outputsize={TIME_SERIES_SIZE}&apikey={API_KEY}"

response = requests.get(URL)
data = response.json()['Time Series (Daily)']
df = pd.DataFrame.from_dict(data).T
df = df.iloc[::-1]  # Reverse the order
df = df.astype(float)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['4. close'].values.reshape(-1, 1))

training_data_len = int(np.ceil(len(scaled_data) * 0.95))
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []

sequence_length = 60
for i in range(sequence_length, len(train_data)):
    x_train.append(train_data[i-sequence_length:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Model Definition and Training
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Creating Testing Dataset and Predicting Prices
test_data = scaled_data[training_data_len - sequence_length:, :]
x_test, y_test = [], scaled_data[training_data_len:, :]

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculating MAPE
mape = mean_absolute_percentage_error(df['4. close'].iloc[training_data_len:].values, predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("QQQ Stock Price Prediction"),
    html.P(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%"),
    dcc.Graph(id='graph'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

# Predicting into the Future (120 units)
future_units = 120
current_input = x_test[-1].reshape((1, sequence_length, 1))  # Taking the last sequence of known output
forecasted_output = []

for i in range(future_units):
    # Predict the next future unit
    next_output = model.predict(current_input)
    forecasted_output.append(next_output[0])
    
    # Update the current_input with the predicted value
    current_input = np.roll(current_input, -1)
    current_input[0, -1, 0] = next_output

# Inverse transform the forecasted output to original scale
forecasted_output_original_scale = scaler.inverse_transform(np.array(forecasted_output).reshape(-1, 1))

# Generating future timestamps
last_timestamp = pd.to_datetime(df.index[-1])
future_timestamps = [last_timestamp + datetime.timedelta(days=i) for i in range(1, future_units+1)]

# Append future timestamps and forecasted_output to the plot
future_df = pd.DataFrame(forecasted_output_original_scale, index=future_timestamps, columns=['Predicted Future'])
concat_df = pd.concat([df, future_df], axis=0)

# Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("QQQ Stock Price Prediction"),
    html.P(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%"),
    dcc.Graph(id='graph'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

# Callback
@app.callback(Output('graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(n):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=concat_df.index, y=concat_df['4. close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=concat_df.index[training_data_len:], y=predictions[:,0], mode='lines', name='Predicted Prices'))
    fig.add_trace(go.Scatter(x=future_timestamps, y=forecasted_output_original_scale[:,0], mode='lines', name='Future Predictions'))
    
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
