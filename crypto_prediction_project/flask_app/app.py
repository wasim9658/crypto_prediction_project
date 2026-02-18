from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
import os

app = Flask(__name__)

# -------- Load model --------
MODEL_PATH = os.path.join("..", "models", "ath_cryptonet.keras")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

# -------- Load scalers --------
feature_scaler = pickle.load(open("../models/feature_scaler.pkl", "rb"))
target_scaler = pickle.load(open("../models/target_scaler.pkl", "rb"))

# -------- Route --------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    trend = None

    if request.method == 'POST':
        try:
            # Get input values
            values = [float(x) for x in request.form.values()]

            # Scale features
            scaled = feature_scaler.transform([values])

            # Create 10-step sequence for LSTM
            data = np.repeat(scaled, 10, axis=0).reshape(1, 10, 5)

            # Model prediction
            scaled_pred = model.predict(data)[0][0]

            # Inverse scale to real BTC price
            prediction = float(
                target_scaler.inverse_transform([[scaled_pred]])[0][0]
            )

            # Trend detection
            current_close = values[0]
            if prediction > current_close:
                trend = "📈 UP"
            else:
                trend = "📉 DOWN"

        except Exception as e:
            prediction = f"Error: {e}"
            trend = None

    return render_template(
        'index.html',
        prediction=prediction,
        trend=trend
    )

# -------- Run --------
if __name__ == "__main__":
    app.run(debug=True)
