import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Opcjonalnie Keras Tuner (jeśli nie masz, tuning pomijamy)
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Opcjonalnie Keras Tuner
try:
    import keras_tuner as kt  # type: ignore
except Exception:
    kt = None


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass(frozen=True)
class TrainConfig:
    arch: str
    epochs: int
    batch_size: int
    learning_rate: float
    augment: bool
    model_out: str
    metrics_out: str
    seed: int = 42


# -------------------------
# Dane (TFDS)
# -------------------------
def preprocess_example(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # TFDS: (28,28,1) uint8
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(0.1, 0.1, name="aug_translate"),
            tf.keras.layers.RandomRotation(0.05, name="aug_rotate"),
            tf.keras.layers.RandomZoom(0.1, name="aug_zoom"),
        ],
        name="augmentation",
    )


def load_datasets(batch_size: int, seed: int):
    (ds_train, ds_test), _ = tfds.load(
        "fashion_mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


# -------------------------
# Modele (2 architektury)
# gotowe pod Keras Tunera
# -------------------------
def build_dense_model(
    input_shape: tuple[int, int, int],
    learning_rate: float,
    augment: bool,
    hp: Optional[object] = None,
) -> tf.keras.Model:
    units1 = 256
    units2 = 128
    dropout = 0.2

    if hp is not None:
        units1 = hp.Int("dense_units1", 128, 512, step=64)
        units2 = hp.Int("dense_units2", 64, 256, step=64)
        dropout = hp.Float("dense_dropout", 0.0, 0.5, step=0.1)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=input_shape, name="input")]

    if augment:
        layers.append(make_augmentation_layer())

    layers.extend(
        [
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(units1, activation="relu", kernel_initializer="he_normal", name="dense_1"),
            tf.keras.layers.Dropout(dropout, name="dropout_1"),
            tf.keras.layers.Dense(units2, activation="relu", kernel_initializer="he_normal", name="dense_2"),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )

    model = tf.keras.Sequential(layers, name="fashion_dense")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model(
    input_shape: tuple[int, int, int],
    learning_rate: float,
    augment: bool,
    hp: Optional[object] = None,
) -> tf.keras.Model:
    filters1 = 32
    filters2 = 64
    dropout = 0.25

    if hp is not None:
        filters1 = hp.Choice("cnn_filters1", [16, 32, 64])
        filters2 = hp.Choice("cnn_filters2", [32, 64, 128])
        dropout = hp.Float("cnn_dropout", 0.0, 0.5, step=0.1)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=input_shape, name="input")]

    if augment:
        layers.append(make_augmentation_layer())

    layers.extend(
        [
            tf.keras.layers.Conv2D(filters1, 3, activation="relu", padding="same", name="conv_1"),
            tf.keras.layers.MaxPool2D(name="pool_1"),
            tf.keras.layers.Conv2D(filters2, 3, activation="relu", padding="same", name="conv_2"),
            tf.keras.layers.MaxPool2D(name="pool_2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dropout(dropout, name="dropout"),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", name="dense"),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )

    model = tf.keras.Sequential(layers, name="fashion_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(
    arch: str,
    input_shape: tuple[int, int, int],
    learning_rate: float,
    augment: bool,
    hp: Optional[object] = None,
) -> tf.keras.Model:
    if arch == "dense":
        return build_dense_model(input_shape, learning_rate, augment, hp=hp)
    if arch == "cnn":
        return build_cnn_model(input_shape, learning_rate, augment, hp=hp)
    raise ValueError("arch must be 'dense' or 'cnn'")


# -------------------------
# Metryki: loss + accuracy + confusion matrix
# -------------------------
def confusion_matrix_from_model(model: tf.keras.Model, ds_test: tf.data.Dataset) -> np.ndarray:
    y_true_all = []
    y_pred_all = []
    for x_batch, y_batch in ds_test:
        probs = model.predict(x_batch, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        y_true_all.append(y_batch.numpy())
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10).numpy()
    return cm


def save_metrics(metrics_path: str, test_loss: float, test_acc: float, cm: np.ndarray):
    payload = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# -------------------------
# Trening
# -------------------------
def train(cfg: TrainConfig):
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ds_train, ds_test = load_datasets(cfg.batch_size, cfg.seed)
    input_shape = (28, 28, 1)

    model = build_model(cfg.arch, input_shape, cfg.learning_rate, cfg.augment)

    model.fit(ds_train, epochs=cfg.epochs, verbose=1)

    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    cm = confusion_matrix_from_model(model, ds_test)

    model.save(cfg.model_out)
    save_metrics(cfg.metrics_out, test_loss, test_acc, cm)

    print(f"\nZapisano model: {cfg.model_out}")
    print(f"Zapisano metryki: {cfg.metrics_out}")
    print(f"TEST loss={test_loss:.4f}, TEST acc={test_acc:.4f}")
    print("Confusion matrix (10x10) zapisano do JSON.")


# -------------------------
# Keras Tuner (opcjonalnie)
# -------------------------
def tune_and_save(cfg: TrainConfig, max_trials: int):
    if kt is None:
        raise RuntimeError("Brak keras-tuner. Zainstaluj: pip install keras-tuner")

    ds_train, ds_test = load_datasets(cfg.batch_size, cfg.seed)
    input_shape = (28, 28, 1)

    # wygodnie dla tunera: numpy
    x_train, y_train = [], []
    for xb, yb in ds_train:
        x_train.append(xb.numpy())
        y_train.append(yb.numpy())
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = [], []
    for xb, yb in ds_test:
        x_test.append(xb.numpy())
        y_test.append(yb.numpy())
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    def hypermodel(hp):
        return build_model(cfg.arch, input_shape, cfg.learning_rate, cfg.augment, hp=hp)

    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        directory="kt_logs",
        project_name=f"fashion_{cfg.arch}",
        overwrite=True,
    )

    stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    tuner.search(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[stop],
        verbose=1,
    )

    best_model = tuner.get_best_models(1)[0]
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)

    probs = best_model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=10).numpy()

    best_model.save(cfg.model_out)
    save_metrics(cfg.metrics_out, test_loss, test_acc, cm)

    print("\nNajlepszy model zapisany.")
    print(f"TEST loss={test_loss:.4f}, TEST acc={test_acc:.4f}")
    best_model.summary()


# -------------------------
# Predykcja z pliku obrazka
# POPRAWKA: auto-invert + debug_preprocessed.png + top-k
# -------------------------
def load_image_for_fashion_mnist(
    path: str,
    invert: Optional[bool] = None,  # None = AUTO
    save_debug_path: Optional[str] = None,
) -> np.ndarray:
    img = Image.open(path).convert("L")   # grayscale
    img = img.resize((28, 28))

    arr = np.array(img, dtype=np.float32)  # 0..255

    # AUTO-INVERT:
    # Jeśli obraz jest ogólnie jasny, to zwykle tło jasne / obiekt ciemny -> robimy negatyw,
    # żeby uzyskać "czarne tło + jasny obiekt" jak w Fashion-MNIST.
    if invert is None:
        invert = (arr.mean() > 127.0)

    if invert:
        arr = 255.0 - arr

    arr = arr / 255.0  # 0..1

    if save_debug_path is not None:
        debug_img = (arr * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(debug_img, mode="L").save(save_debug_path)

    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    return arr


def predict_image(model_path: str, image_path: str, invert: Optional[bool], debug: bool, topk: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu: {model_path}. Najpierw uruchom trening.")

    model = tf.keras.models.load_model(model_path)

    debug_path = "debug_preprocessed.png" if debug else None
    x = load_image_for_fashion_mnist(image_path, invert=invert, save_debug_path=debug_path)
    x = np.expand_dims(x, axis=0)  # (1,28,28,1)

    probs = model.predict(x, verbose=0)[0]  # (10,)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    print(f"Klasa: {idx} ({CLASS_NAMES[idx]})")
    print(f"Pewność: {conf:.4f}")

    if topk > 1:
        order = np.argsort(probs)[::-1][:topk]
        print("\nTOP predykcje:")
        for j in order:
            print(f"  {int(j)} ({CLASS_NAMES[int(j)]}): {float(probs[j]):.4f}")

    if debug:
        print("\nZapisano podgląd tego co widzi model do: debug_preprocessed.png")
        print("Jeśli ten obraz wygląda jak plama/tło dominuje -> wynik może być losowy.")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Lab 5 - Fashion MNIST (train + predict)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Trening modelu i zapis do .keras + metryki")
    p_train.add_argument("--arch", choices=["dense", "cnn"], default="cnn")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--augment", action="store_true")
    p_train.add_argument("--model_out", default="fashion_model.keras")
    p_train.add_argument("--metrics_out", default="metrics.json")

    p_tune = sub.add_parser("tune", help="Keras Tuner (RandomSearch) + zapis najlepszego modelu")
    p_tune.add_argument("--arch", choices=["dense", "cnn"], default="cnn")
    p_tune.add_argument("--epochs", type=int, default=10)
    p_tune.add_argument("--batch_size", type=int, default=64)
    p_tune.add_argument("--lr", type=float, default=1e-3)
    p_tune.add_argument("--augment", action="store_true")
    p_tune.add_argument("--max_trials", type=int, default=10)
    p_tune.add_argument("--model_out", default="fashion_tuned.keras")
    p_tune.add_argument("--metrics_out", default="tuned_metrics.json")

    p_pred = sub.add_parser("predict", help="Klasyfikacja obrazka z pliku")
    p_pred.add_argument("--model", required=True, help="Ścieżka do .keras")
    p_pred.add_argument("--image", required=True, help="Ścieżka do obrazka")
    p_pred.add_argument("--invert", action="store_true", help="Wymuś negatyw (inwersję)")
    p_pred.add_argument("--no_invert", action="store_true", help="Wyłącz negatyw (brak inwersji)")
    p_pred.add_argument("--debug", action="store_true", help="Zapisz debug_preprocessed.png")
    p_pred.add_argument("--topk", type=int, default=3, help="Pokaż top-k klas (domyślnie 3)")

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            augment=args.augment,
            model_out=args.model_out,
            metrics_out=args.metrics_out,
        )
        train(cfg)
        return

    if args.cmd == "tune":
        cfg = TrainConfig(
            arch=args.arch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            augment=args.augment,
            model_out=args.model_out,
            metrics_out=args.metrics_out,
        )
        tune_and_save(cfg, max_trials=args.max_trials)
        return

    if args.cmd == "predict":
        inv: Optional[bool] = None
        if args.invert and args.no_invert:
            print("Nie używaj jednocześnie --invert i --no_invert")
            return
        if args.invert:
            inv = True
        if args.no_invert:
            inv = False

        predict_image(args.model, args.image, invert=inv, debug=args.debug, topk=args.topk)
        return


if __name__ == "__main__":
    main()


#Trening danse
# python main.py train --arch dense --epochs 10 --batch_size 64 --lr 0.001 --model_out dense.keras --metrics_out dense_metrics.json
# cnn
# python main.py train --arch cnn --epochs 10 --batch_size 64 --lr 0.001 --model_out cnn.keras --metrics_out cnn_metrics.json
# augmentacja
# python main.py train --arch cnn --augment --epochs 10 --model_out cnn_aug.keras --metrics_out cnn_aug_metrics.json
# python main.py predict --model cnn.keras --image sample.png
# keras Tuner