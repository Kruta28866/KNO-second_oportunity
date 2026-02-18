import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Keras Tuner (pakiet: keras-tuner)
import keras_tuner as kt


DATA_PATH = "wine.data"
BASELINE_MODEL_PATH = "baseline_model.keras"
TUNED_MODEL_PATH = "tuned_best_model.keras"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    val_split: float = 0.2
    seed: int = 42


# =========================
# 1) Dane: wczytanie + shuffle
# =========================
def load_data(path: str, seed: int):
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

    # shuffle (ważne)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X = df.drop("class", axis=1).values.astype(np.float32)
    y = df["class"].values.astype(np.int32) - 1  # 0,1,2

    return X, y


def split_data(X, y, seed: int):
    # train/test (walidacja będzie z train przez validation_split)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# =========================
# 2) Normalization layer
# =========================
def make_normalizer(X_train: np.ndarray) -> tf.keras.layers.Normalization:
    norm = tf.keras.layers.Normalization(name="normalization")
    norm.adapt(X_train)  # uczy się średnich/odchyleń z TRAIN
    return norm


# =========================
# 3) Baseline model (z Lab 3) + opcjonalna normalizacja
# =========================
def build_baseline_model(
    input_dim: int,
    learning_rate: float,
    use_normalization: bool,
    normalizer: tf.keras.layers.Normalization | None,
) -> tf.keras.Model:
    layers = []

    layers.append(tf.keras.layers.Input(shape=(input_dim,), name="input"))

    if use_normalization:
        if normalizer is None:
            raise ValueError("use_normalization=True, ale normalizer=None")
        layers.append(normalizer)

    # "Model 2" styl: 64 -> dropout -> 32 -> softmax
    layers.extend(
        [
            tf.keras.layers.Dense(
                64, activation="tanh", kernel_initializer="glorot_uniform", name="hidden_1"
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(
                32, activation="tanh", kernel_initializer="glorot_uniform", name="hidden_2"
            ),
            tf.keras.layers.Dense(3, activation="softmax", name="output"),
        ]
    )

    model = tf.keras.Sequential(layers, name="Baseline_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",  # bo y to 0/1/2, bez one-hot
        metrics=["accuracy"],
    )
    return model


# =========================
# 4) Metryki + confusion matrix
# =========================
def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm


# =========================
# 5) Keras Tuner - funkcja budująca model z hp
#    3 parametry: learning_rate (wymagany), units_1, dropout_rate
# =========================
def build_model_for_tuner(hp: kt.HyperParameters, input_dim: int, normalizer):
    lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)

    layers = [
        tf.keras.layers.Input(shape=(input_dim,), name="input"),
        normalizer,
        tf.keras.layers.Dense(units_1, activation="relu", kernel_initializer="he_normal", name="hidden_1"),
        tf.keras.layers.Dropout(dropout_rate, name="dropout"),
        tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal", name="hidden_2"),
        tf.keras.layers.Dense(3, activation="softmax", name="output"),
    ]

    model = tf.keras.Sequential(layers, name="Tuned_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# 6) Uruchomienie tunera
# =========================
def run_tuner(X_train, y_train, input_dim: int, normalizer, config: TrainConfig):
    tuner = kt.Hyperband(
        hypermodel=lambda hp: build_model_for_tuner(hp, input_dim, normalizer),
        objective="val_accuracy",
        max_epochs=40,
        factor=3,
        directory="kt_logs",
        project_name="wine_tuning",
        overwrite=True,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)

    tuner.search(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.val_split,
        callbacks=[stop_early],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    return best_hp, best_model


# =========================
# 7) MAIN: tryby
# =========================
def main():
    parser = argparse.ArgumentParser(description="Lab 4: Wine - baseline + normalization + keras tuner")
    parser.add_argument("--baseline", action="store_true", help="Uruchom baseline bez normalizacji")
    parser.add_argument("--baseline_norm", action="store_true", help="Uruchom baseline z normalizacją")
    parser.add_argument("--tune", action="store_true", help="Uruchom Keras Tuner (z normalizacją)")
    parser.add_argument("--summary", action="store_true", help="Pokaż summary zapisanych modeli")
    args = parser.parse_args()

    config = TrainConfig()

    if not os.path.exists(DATA_PATH):
        print(f"Brak {DATA_PATH}. Pobierz wine.data i wrzuć do folderu.")
        return

    # dane
    X, y = load_data(DATA_PATH, seed=config.seed)
    X_train, X_test, y_train, y_test = split_data(X, y, seed=config.seed)
    input_dim = X_train.shape[1]

    # normalizer (uczony tylko na train)
    normalizer = make_normalizer(X_train)

    # 1) Baseline bez norm
    if args.baseline:
        model = build_baseline_model(
            input_dim=input_dim,
            learning_rate=0.001,
            use_normalization=False,
            normalizer=None,
        )

        print("Baseline (bez normalizacji):")
        print(f"epochs={config.epochs}, batch_size={config.batch_size}, lr=0.001")

        model.fit(
            X_train,
            y_train,
            validation_split=config.val_split,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
        )

        acc, cm = evaluate_model(model, X_test, y_test)
        print(f"TEST accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)

        model.save(BASELINE_MODEL_PATH)
        print(f"Zapisano baseline do {BASELINE_MODEL_PATH}")
        return

    # 2) Baseline z norm
    if args.baseline_norm:
        model = build_baseline_model(
            input_dim=input_dim,
            learning_rate=0.001,
            use_normalization=True,
            normalizer=normalizer,
        )

        print("Baseline (z normalizacją):")
        print(f"epochs={config.epochs}, batch_size={config.batch_size}, lr=0.001")

        model.fit(
            X_train,
            y_train,
            validation_split=config.val_split,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0,
        )

        acc, cm = evaluate_model(model, X_test, y_test)
        print(f"TEST accuracy: {acc:.4f}")
        print("Confusion matrix:\n", cm)

        model.save(BASELINE_MODEL_PATH)
        print(f"Zapisano baseline do {BASELINE_MODEL_PATH}")
        return

    # 3) Tuning
    if args.tune:
        print("Tuning (Hyperband) + normalizacja")
        print(f"epochs={config.epochs}, batch_size={config.batch_size}")

        best_hp, best_model = run_tuner(X_train, y_train, input_dim, normalizer, config)

        print("Najlepsze hiperparametry:")
        for k in best_hp.values.keys():
            print(f"  {k}: {best_hp.get(k)}")

        acc, cm = evaluate_model(best_model, X_test, y_test)
        print(f"TEST accuracy (best tuned): {acc:.4f}")
        print("Confusion matrix:\n", cm)

        print("\nModel summary:")
        best_model.summary()

        best_model.save(TUNED_MODEL_PATH)
        print(f"Zapisano tuned model do {TUNED_MODEL_PATH}")
        return

    # 4) Summary zapisanych modeli
    if args.summary:
        if os.path.exists(BASELINE_MODEL_PATH):
            print("\n=== BASELINE MODEL SUMMARY ===")
            m = tf.keras.models.load_model(BASELINE_MODEL_PATH)
            m.summary()
        else:
            print(f"Brak {BASELINE_MODEL_PATH}")

        if os.path.exists(TUNED_MODEL_PATH):
            print("\n=== TUNED MODEL SUMMARY ===")
            m = tf.keras.models.load_model(TUNED_MODEL_PATH)
            m.summary()
        else:
            print(f"Brak {TUNED_MODEL_PATH}")
        return

    print("Wybierz tryb:")
    print("  python main.py --baseline")
    print("  python main.py --baseline_norm")
    print("  python main.py --tune")
    print("  python main.py --summary")


if __name__ == "__main__":
    main()

# python main.py --baseline
# python main.py --baseline_norm
# python main.py --tune
# python main.py --summary
