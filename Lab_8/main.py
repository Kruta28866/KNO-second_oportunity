import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -----------------------
# Parametry zadania
# -----------------------
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

TOTAL_POINTS = 6000        # ile punktów funkcji generujemy
LOOKBACK = 60              # ile punktów wejścia (okno)
N_AHEAD = 10               # przewidujemy wartość po N krokach
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 25

# -----------------------
# 1) Generacja funkcji + opcjonalny szum
# -----------------------
# Oś X jako równomierna siatka
x = np.linspace(0, 60 * np.pi, TOTAL_POINTS).astype(np.float32)

# Funkcja klasyczna: sinus + lekka domieszka harmonicznej
y = (np.sin(x) + 0.3 * np.sin(3 * x)).astype(np.float32)

# Możesz dodać mały szum, żeby model nie był "za łatwy"
noise = 0.02 * np.random.randn(TOTAL_POINTS).astype(np.float32)
y_noisy = y + noise

# Normalizacja (tu prosta: średnia=0, std=1)
mean = y_noisy.mean()
std = y_noisy.std() + 1e-8
y_norm = (y_noisy - mean) / std

# -----------------------
# 2) Budowa zbioru (X: okno LOOKBACK, target: y[t+N_AHEAD])
# -----------------------
def make_dataset(series: np.ndarray, lookback: int, n_ahead: int):
    X, Y = [], []
    # ostatni indeks startu okna:
    # start + lookback - 1 to ostatni element wejścia
    # target = start + lookback - 1 + n_ahead
    last_start = len(series) - lookback - n_ahead
    for start in range(last_start):
        window = series[start:start + lookback]
        target = series[start + lookback - 1 + n_ahead]
        X.append(window)
        Y.append(target)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    # LSTM oczekuje (batch, timesteps, features)
    X = X[..., np.newaxis]  # features=1
    return X, Y

X, Y = make_dataset(y_norm, LOOKBACK, N_AHEAD)

# Split train/val
n_train = int(len(X) * TRAIN_RATIO)
X_train, y_train = X[:n_train], Y[:n_train]
X_val, y_val = X[n_train:], Y[n_train:]

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)

# -----------------------
# 3) Model LSTM
# -----------------------
model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.1),
    layers.LSTM(32),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)  # jedna wartość: y po N krokach
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    callbacks=callbacks
)

# -----------------------
# 4) Ewaluacja + wykresy
# -----------------------
# Predykcje na walidacji
pred_val = model.predict(X_val, verbose=0).squeeze()

# Denormalizacja do skali oryginalnej
y_val_den = y_val * std + mean
pred_val_den = pred_val * std + mean

# Wykres krzywych uczenia
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Uczenie: loss")
plt.xlabel("Epoka")
plt.ylabel("MSE")
plt.legend()
plt.show()

# Porównanie: kilka przykładowych predykcji vs prawda
plt.figure(figsize=(10, 4))
k = 400  # ile punktów walidacji pokazać
plt.plot(y_val_den[:k], label="prawda")
plt.plot(pred_val_den[:k], label=f"predykcja (+{N_AHEAD})")
plt.title("Predykcja wartości funkcji po N krokach (walidacja)")
plt.xlabel("indeks próbki (na zbiorze val)")
plt.ylabel("y")
plt.legend()
plt.show()

# -----------------------
# 5) Pokaz działania "na osi czasu" (opcjonalnie, fajne do prezentacji)
# -----------------------
# Bierzemy fragment końcówki sygnału i pokazujemy:
# - ostatnie LOOKBACK wartości
# - prawdziwą wartość po N krokach
# - predykcję po N krokach
idx = len(X_val) - 1
window = X_val[idx:idx+1]  # shape (1, LOOKBACK, 1)
pred_one = model.predict(window, verbose=0).item()

# denormalizacja
window_den = window.squeeze() * std + mean
pred_one_den = pred_one * std + mean
true_one_den = y_val[idx] * std + mean

plt.figure(figsize=(10, 4))
plt.plot(np.arange(LOOKBACK), window_den, label="wejście (okno)")
plt.scatter([LOOKBACK - 1 + N_AHEAD], [true_one_den], label="prawda po N", marker="x", s=80)
plt.scatter([LOOKBACK - 1 + N_AHEAD], [pred_one_den], label="pred po N", marker="o", s=60)
plt.title("Jedna predykcja po N krokach na fragmencie sygnału")
plt.xlabel("czas w obrębie okna + przesunięcie")
plt.ylabel("y")
plt.legend()
plt.show()
