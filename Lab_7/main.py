#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def read_close_csv(path: str, close_col: str | None = None, date_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')

    print(df.head())

    # date handling (optional)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    # close column detection
    df = df.dropna(subset=['<CLOSE>']).copy()
    df = df.rename(columns={'<CLOSE>': "close"}).reset_index(drop=True)

    return df[["close"]]


def make_windows(series_scaled: np.ndarray, lookback: int):
    """
    series_scaled: shape (N, 1)
    X: (N-lookback, lookback, 1)
    y: (N-lookback, 1)   next-step
    """
    X, y = [], []
    for i in range(len(series_scaled) - lookback):
        X.append(series_scaled[i:i + lookback])
        y.append(series_scaled[i + lookback])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_split_time(X, y, val_ratio: float):
    n = len(X)
    n_val = max(1, int(n * val_ratio))
    return (X[:-n_val], y[:-n_val]), (X[-n_val:], y[-n_val:])


def build_lstm(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def forecast_autoregressive(model, last_window_scaled: np.ndarray, n: int):
    """
    last_window_scaled: (lookback, 1) scaled
    returns scaled preds: (n, 1)
    """
    preds = []
    w = last_window_scaled.copy()
    for _ in range(n):
        yhat = model.predict(w[None, ...], verbose=0)[0]  # (1,)
        preds.append(yhat)
        w = np.vstack([w[1:], yhat.reshape(1, 1)])
    return np.array(preds, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Simple Close-only LSTM forecaster")
    ap.add_argument("--history", required=True, help="CSV with historical data")
    ap.add_argument("--result", required=True, help="Output CSV for predictions")
    ap.add_argument("--n", type=int, required=True, help="Number of future steps to forecast")
    ap.add_argument("--close_col", default=None, help="Column name for Close (default: auto-detect)")
    ap.add_argument("--date_col", default=None, help="Optional date column to sort by (e.g., Date)")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--outdir", default="out_simple")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read Close-only series
    df = read_close_csv(args.history, close_col=args.close_col, date_col=args.date_col)
    close = df["close"].values.astype(np.float32).reshape(-1, 1)

    if len(close) <= args.lookback + 5:
        raise ValueError("Not enough data for chosen lookback. Provide more history or reduce --lookback.")

    # Scale
    scaler = StandardScaler()
    close_scaled = scaler.fit_transform(close)

    # Windows
    X, y = make_windows(close_scaled, args.lookback)
    (X_train, y_train), (X_val, y_val) = train_val_split_time(X, y, args.val_ratio)

    # Model
    model = build_lstm(input_shape=X_train.shape[1:])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(args.outdir, "model.keras"),
                                        monitor="val_loss", save_best_only=True),
    ]
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
        callbacks=callbacks
    )

    # Forecast
    last_window = close_scaled[-args.lookback:]  # (lookback, 1)
    preds_scaled = forecast_autoregressive(model, last_window, args.n)  # (n, 1)
    preds = scaler.inverse_transform(preds_scaled).reshape(-1)

    # Save predictions
    out_df = pd.DataFrame({"pred_close": preds})
    out_df.to_csv(args.result, index=False)

    # Optional plots
    if args.plots:
        plt.figure()
        plt.plot(hist.history["loss"], label="train_loss")
        plt.plot(hist.history["val_loss"], label="val_loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "training.png"))
        plt.close()

        tail = close.reshape(-1)[-200:]
        plt.figure()
        plt.plot(np.arange(len(tail)), tail, label="history")
        plt.plot(np.arange(len(tail), len(tail) + len(preds)), preds, label="forecast")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "forecast.png"))
        plt.close()

    # Save simple metrics
    best_val = float(np.min(hist.history["val_loss"]))
    best_mae = float(np.min(hist.history["val_mae"]))
    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_val_loss={best_val}\n")
        f.write(f"best_val_mae={best_mae}\n")
        f.write(f"tf_version={tf.__version__}\n")
        f.write(f"gpus={[str(g) for g in tf.config.list_physical_devices('GPU')]}\n")

    print(f"OK. Saved predictions to: {args.result}")
    print(f"Artifacts in: {args.outdir}")


if __name__ == "__main__":
    main()