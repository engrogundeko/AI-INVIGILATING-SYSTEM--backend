import numpy as np
import os
import json
import pandas as pd

def create_lstm_dataset(flows, sequence_length):
    X, y = [], []
    for i in range(len(flows) - sequence_length):
        X.append(flows[i:i + sequence_length])  # The sequence of values
        y.append(flows[i + sequence_length])    # The next value (target)
    return np.array(X), np.array(y)

# Load flow values (example data)
with open("normalized1.json", "r") as json_file:
    data = json.load(json_file)

# Define the sequence length (for LSTM input)
sequence_length = 10

# Number of values per second
values_per_second = 6
time_interval = 1 / values_per_second  # Interval between values

# Create a time index (assuming the first value is at time 0)
# num_values = len(data)
# time_index = pd.date_range(start='2024-01-01 00:00:00', periods=num_values, freq=f'{int(1000 * time_interval)}ms')

# Create a time series DataFrame
# df = pd.DataFrame({'Timestamp': time_index, 'Optical_Flow': data})

# Create the dataset
X, y = create_lstm_dataset(data, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(X[:6])


# Save to a file
# with open("lstm_dataset", "w") as json_file:
#     dt = dict(X=X, y=y)
#     json.dump(dt, json_file, indent=4)
np.savez("lstm1.npz", X=X, y=y)
# np.load

# print("Dataset saved successfully.")
