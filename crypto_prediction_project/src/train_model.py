import yfinance as yf
import pandas as pd
import numpy as np
import ta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Create models folder
os.makedirs("../models", exist_ok=True)

# ---------- Load data ----------

df = yf.download("BTC-USD", period="3y", interval="1d")

# ✅ FIX: flatten MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ✅ safety check
print("Columns:", df.columns)

df.dropna(inplace=True)


# ---------- Technical indicators ----------
df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['macd'] = ta.trend.MACD(df['Close']).macd()
df['volatility'] = df['Close'].rolling(10).std()
df.dropna(inplace=True)

# ---------- Dummy sentiment ----------
np.random.seed(42)
df['sentiment'] = np.random.uniform(-1,1,len(df))

features = ['Close','rsi','macd','volatility','sentiment']
X = df[features]
y = df['Close']

# ---------- Scaling ----------
# ---------- Feature Scaling ----------
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

# ---------- Target Scaling ----------
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y.values.reshape(-1,1))

# save scalers
pickle.dump(feature_scaler, open("../models/feature_scaler.pkl","wb"))
pickle.dump(target_scaler, open("../models/target_scaler.pkl","wb"))


# ---------- Sequence ----------
def create_seq(X,y,seq=10):
    Xs, ys = [], []
    for i in range(len(X)-seq):
        Xs.append(X[i:i+seq])
        ys.append(y.iloc[i+seq])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_seq(pd.DataFrame(X_scaled), pd.Series(y_scaled.flatten()))

# ---------- Attention ----------
from tensorflow.keras import layers, models

def attention_layer(x):
    score = layers.Dense(1, activation='tanh')(x)
    weights = layers.Softmax(axis=1)(score)

    context = layers.Multiply()([weights, x])

    # ✅ SAFE reduction (NO Lambda)
    context = layers.GlobalAveragePooling1D()(context)

    return context

def build_model(shape):
    inp = layers.Input(shape=shape)

    # BiLSTM layer
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(inp)

    # Dropout to prevent overfitting
    x = layers.Dropout(0.2)(x)

    # Attention
    x = attention_layer(x)

    # Dense layers (stronger learning)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)

    # Output layer
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)

    # Better compile settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


model = build_model(X_seq.shape[1:])
history = model.fit(X_seq, y_seq, epochs=15, batch_size=32)

# ---------- Train test split ----------
split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ---------- Save model ----------
model.save("../models/ath_cryptonet.keras")
print("✅ Model saved!")
