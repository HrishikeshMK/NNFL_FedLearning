from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from flask_cors import CORS

# Create the Flask app
app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("fraud_detection_cnn_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data
        input_data = request.json.get("data", [])
        if len(input_data) != 30:
            return jsonify({"error": "Invalid input data"}), 400
        
        # Convert input to a numpy array and reshape for prediction
        input_array = np.array(input_data, dtype=np.float32).reshape(1, -1, 1, 1)
        
        # Make prediction
        prediction_probs = model.predict(input_array)
        prediction = int(prediction_probs[0, 0] > 0.5)
        probability = float(prediction_probs[0, 0])
        
        # Return the result
        return jsonify({"prediction": prediction, "probability": probability})
    
    except Exception as e:
       return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
