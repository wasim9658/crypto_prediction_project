import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
import yfinance as yf
import ta

globals()["tf"] = tf


# ---------- Paths ----------
MODEL_PATH = "../models/ath_cryptonet.keras"
FEATURE_SCALER_PATH = "../models/feature_scaler.pkl"
TARGET_SCALER_PATH = "../models/target_scaler.pkl"

os.makedirs("../results", exist_ok=True)

# ---------- Load model ----------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

feature_scaler = pickle.load(open(FEATURE_SCALER_PATH, "rb"))
target_scaler = pickle.load(open(TARGET_SCALER_PATH, "rb"))

# ---------- Load data ----------
df = yf.download("BTC-USD", period="3y", interval="1d")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['macd'] = ta.trend.MACD(df['Close']).macd()
df['volatility'] = df['Close'].rolling(10).std()

np.random.seed(42)
df['sentiment'] = np.random.uniform(-1, 1, len(df))

df.dropna(inplace=True)

features = ['Close','rsi','macd','volatility','sentiment']
X = df[features]
y = df['Close']

# ---------- Scale ----------
X_scaled = feature_scaler.transform(X)
y_scaled = target_scaler.transform(y.values.reshape(-1,1))

# ---------- Sequence ----------
def create_seq(X, y, seq=10):
    Xs, ys = [], []
    for i in range(len(X)-seq):
        Xs.append(X[i:i+seq])
        ys.append(y[i+seq])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_seq(X_scaled, y_scaled.flatten())

# ---------- Split ----------
split = int(len(X_seq) * 0.8)
X_test = X_seq[split:]
y_test = y_seq[split:]

# ---------- Predict ----------
pred_scaled = model.predict(X_test)

# inverse scale
y_pred = target_scaler.inverse_transform(pred_scaled)
y_true = target_scaler.inverse_transform(y_test.reshape(-1,1))

# ---------- Plot ----------
plt.figure(figsize=(10,5))
plt.plot(y_true[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.title("ATH-CryptoNet: Actual vs Predicted BTC Price")
plt.xlabel("Time")
plt.ylabel("BTC Price")
plt.legend()

# ---------- Save ----------
plot_path = "../results/prediction_vs_actual.png"
plt.savefig(plot_path)
plt.show()

print("✅ Plot saved to:", plot_path)
