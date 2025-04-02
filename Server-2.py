import flwr as fl
import tensorflow as tf

# Define the global model (same as client-side model)
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Flower strategy for Federated Averaging
class SaveableStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            self.model.set_weights(aggregated_weights)
            self.model.save(f"fraud_detection_model.h5")
            print(f"Saved global model at round {rnd}")
        return aggregated_weights

# Main logic for starting the server
if __name__ == "__main__":
    input_dim = 30  # Number of features in the dataset
    global_model = create_model(input_dim)

    # Start Flower server with custom strategy
    strategy = SaveableStrategy(global_model)
    fl.server.start_server(server_address="localhost:8080",config = fl.server.ServerConfig(num_rounds=3), strategy=strategy)
