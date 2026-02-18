import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


DATA_PATH = "wine.data"
MODEL_PATH = "best_model.keras"

def load_and_prepare_data(path: str):
    column_names = [
        "class",
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity",
        "magnesium",
        "phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315",
        "proline",
    ]

    df = pd.read_csv(path, header=None, names=column_names)

    # shuffle (mieszamy wiersze)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    X = df.drop("class", axis=1).values.astype(np.float32)

    y = df["class"].values.astype(np.int32) - 1

    # one-hot
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_model_1(input_dim: int, learning_rate: float):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input"),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_initializer="he_normal",
                name="hidden_1",
            ),
            tf.keras.layers.Dense(
                16,
                activation="relu",
                kernel_initializer="he_normal",
                name="hidden_2",
            ),
            tf.keras.layers.Dense(3, activation="softmax", name="output"),
        ],
        name="Model_1",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# 3) Model 2
# =========================
def build_model_2(input_dim: int, learning_rate: float):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,), name="input"),
            tf.keras.layers.Dense(
                64,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                name="hidden_1",
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(
                32,
                activation="tanh",
                kernel_initializer="glorot_uniform",
                name="hidden_2",
            ),
            tf.keras.layers.Dense(3, activation="softmax", name="output"),
        ],
        name="Model_2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# 4) Wykresy
# =========================
def plot_history(history: tf.keras.callbacks.History, title: str):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


# =========================
# 5) Trenowanie + wybór lepszego modelu
# =========================
def train_and_select_best_model(X_train, X_test, y_train, y_test):
    epochs = 50
    batch_size = 16
    learning_rate = 0.001

    print(f"Trening: epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")

    model1 = build_model_1(X_train.shape[1], learning_rate)
    model2 = build_model_2(X_train.shape[1], learning_rate)

    history1 = model1.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    history2 = model2.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    plot_history(history1, "Model 1")
    plot_history(history2, "Model 2")

    test_loss1, test_acc1 = model1.evaluate(X_test, y_test, verbose=0)
    test_loss2, test_acc2 = model2.evaluate(X_test, y_test, verbose=0)

    print(f"Model 1 Test Accuracy: {test_acc1:.4f}")
    print(f"Model 2 Test Accuracy: {test_acc2:.4f}")

    if test_acc1 > test_acc2:
        print("Model 1 wygrywa")
        return model1
    else:
        print("Model 2 wygrywa")
        return model2


# =========================
# 6) Predykcja
# =========================
def predict_class(model: tf.keras.Model, features):
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    probs = model.predict(x, verbose=0)  # shape (1,3)
    predicted_class_0based = int(np.argmax(probs, axis=1)[0])  # 0,1,2
    return predicted_class_0based + 1  # 1,2,3 (jak w oryginalnym dataset)


# =========================
# 7) MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Wine classification (UCI Wine)")
    parser.add_argument("--train", action="store_true", help="Trenuj modele i zapisz najlepszy")
    parser.add_argument("--predict", action="store_true", help="Wykonaj predykcję z podanych cech")
    parser.add_argument(
        "--features",
        nargs=13,
        type=float,
        help="13 cech wina w kolejności jak w dataset (bez klasy)",
    )

    args = parser.parse_args()

    if args.train:
        if not os.path.exists(DATA_PATH):
            print(f"Brak pliku {DATA_PATH}. Pobierz go i wrzuć do folderu.")
            return

        X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_PATH)
        best_model = train_and_select_best_model(X_train, X_test, y_train, y_test)

        best_model.save(MODEL_PATH)
        print(f"Zapisano najlepszy model do: {MODEL_PATH}")
        return

    if args.predict:
        if args.features is None:
            print("Podaj 13 cech: --features <13 liczb>")
            return

        if not os.path.exists(MODEL_PATH):
            print(f"Brak {MODEL_PATH}. Najpierw uruchom trening: python main.py --train")
            return

        model = tf.keras.models.load_model(MODEL_PATH)
        pred = predict_class(model, args.features)
        print(f"Przewidywana klasa wina: {pred}")
        return

    print("Podaj tryb pracy: --train albo --predict")
    print("Przykład treningu: python main.py --train")
    print(
        "Przykład predykcji: python main.py --predict --features "
        "14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065"
    )


if __name__ == "__main__":
    main()

# python main.py --train
# python main.py --predict --features 14.23 1.71 2.43 15.6 127 2.8 3.06 0.28 2.29 5.64 1.04 3.92 1065