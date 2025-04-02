import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the TensorFlow model
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Data preparation
def prepare_data():
    # Load the Kaggle dataset
    df = pd.read_csv("creditcard.csv")
    
    # Separate features and target
    X = df.drop(columns=["Class", "Time"]).values
    y = df["Class"].values
    
    # Split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test

# Flower client class
class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def get_parameters(self):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=1)
        return self.get_parameters(), len(self.x_train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Main logic for starting the client
if __name__ == "__main__":
    # Prepare data
    x_train, y_train, x_test, y_test = prepare_data()
    input_dim = x_train.shape[1]

    # Create model
    model = create_model(input_dim)

    # Start Flower client
    client = FraudDetectionClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
