import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset = pd.read_excel('DataReal.xlsx')

# Define the features and the target
X = dataset.iloc[:, :6].values
y = dataset.iloc[:, -2].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train_scaled, y_train, batch_size=32, epochs=100)

# Save the model
ann.save("saved_model")

# Load the model
model = tf.keras.models.load_model("saved_model")

# Scale the new input data using the same scaler
new_input_variables = np.array([[311, 91.9, -8, 25, 0.07, 0.5]])  # Example input data
new_input_variables_scaled = scaler.transform(new_input_variables)

# Make predictions with the loaded model
predictions = model.predict(new_input_variables_scaled)

# Post-process predictions if needed (e.g., ensure non-negative outputs)
predictions = np.maximum(predictions, 0)

# Print the predictions
print("Predictions:", predictions)
