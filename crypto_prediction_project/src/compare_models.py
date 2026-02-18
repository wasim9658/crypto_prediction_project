import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
import yfinance as yf
import ta

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Fix for Lambda model loading
globals()["tf"] = tf

# ---------- Paths ----------
MODEL_PATH = "../models/ath_cryptonet.keras"
FEATURE_SCALER_PATH = "../models/feature_scaler.pkl"
TARGET_SCALER_PATH = "../models/target_scaler.pkl"

os.makedirs("../results", exist_ok=True)

# ---------- Load deep model ----------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

feature_scaler = pickle.load(open(FEATURE_SCALER_PATH, "rb"))
target_scaler = pickle.load(open(TARGET_SCALER_PATH, "rb"))

# ---------- Load data ----------
df = yf.download("BTC-USD", period="3y", interval="1d")

# flatten columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.dropna(inplace=True)

# ---------- Indicators ----------
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
    for i in range(len(X) - seq):
        Xs.append(X[i:i+seq])
        ys.append(y[i+seq])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_seq(X_scaled, y_scaled)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# ---------- Classical models ----------
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train_flat, y_train)
rf.fit(X_train_flat, y_train)

lr_pred = lr.predict(X_test_flat)
rf_pred = rf.predict(X_test_flat)

# ---------- Deep model ----------
dl_pred = model.predict(X_test)

# ---------- Inverse scale ----------
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

lr_pred_inv = target_scaler.inverse_transform(
    lr_pred.reshape(-1, 1)
)

rf_pred_inv = target_scaler.inverse_transform(
    rf_pred.reshape(-1, 1)
)

dl_pred_inv = target_scaler.inverse_transform(
    dl_pred.reshape(-1, 1)
)

# ---------- Metrics ----------
def evaluate(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred)
    )

lr_mae, lr_rmse, lr_r2 = evaluate(y_test_inv, lr_pred_inv)
rf_mae, rf_rmse, rf_r2 = evaluate(y_test_inv, rf_pred_inv)
dl_mae, dl_rmse, dl_r2 = evaluate(y_test_inv, dl_pred_inv)

# ---------- Results table ----------
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "ATH-CryptoNet (Proposed)"
    ],
    "MAE": [lr_mae, rf_mae, dl_mae],
    "RMSE": [lr_rmse, rf_rmse, dl_rmse],
    "R2 Score": [lr_r2, rf_r2, dl_r2]
})

print("\n📊 Model Comparison:\n")
print(results)

# ---------- Save ----------
csv_path = "../results/model_comparison.csv"
results.to_csv(csv_path, index=False)

print("\n✅ Comparison saved to:", csv_path)
