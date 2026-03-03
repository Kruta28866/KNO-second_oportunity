import os
import argparse
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

from tensorflow import keras
from PIL import Image
import os

(x_train, _), _ = keras.datasets.cifar10.load_data()

out_dir = "dataset/images"
os.makedirs(out_dir, exist_ok=True)

for i in range(50):   # >=20, dajemy 50
    img = Image.fromarray(x_train[i])
    img.save(f"{out_dir}/img_{i:03d}.png")

print("Zapisano CIFAR-10 do dataset/images")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_ds(data_dir, img_size, batch_size, val_split, seed):
    train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=None,
        label_mode=None,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    val = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=None,
        label_mode=None,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)
    train = train.map(lambda x: rescale(tf.cast(x, tf.float32)), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val = val.map(lambda x: rescale(tf.cast(x, tf.float32)), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train, val


def build_models(img_size, latent_dim):
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ]
    )

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
            tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(latent_dim),
        ]
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(8 * 8 * 256, activation="relu"),
            tf.keras.layers.Reshape((8, 8, 256)),
            tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid"),
        ]
    )

    autoencoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
            aug,
            encoder,
            decoder,
        ]
    )

    return autoencoder, encoder, decoder


def save_batch(x, out_dir, prefix, n=16):
    ensure_dir(out_dir)
    x = x[:n]
    for i in range(x.shape[0]):
        tf.keras.utils.save_img(os.path.join(out_dir, f"{prefix}_{i:03d}.png"), x[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="outputs_lab6")
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "recon_train"))
    ensure_dir(os.path.join(args.out_dir, "recon_val"))
    ensure_dir(os.path.join(args.out_dir, "generated"))
    ensure_dir(os.path.join(args.out_dir, "checkpoints"))

    train_ds, val_ds = load_ds(args.data_dir, args.img_size, args.batch_size, args.val_split, args.seed)

    autoencoder, encoder, decoder = build_models(args.img_size, args.latent_dim)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=["mae"],
    )

    autoencoder.fit(
        train_ds.map(lambda x: (x, x)),
        validation_data=val_ds.map(lambda x: (x, x)),
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "checkpoints", "best.keras"), save_best_only=True),
        ],
    )

    train_batch = next(iter(train_ds))
    val_batch = next(iter(val_ds))

    train_recon = autoencoder.predict(train_batch, verbose=0)
    val_recon = autoencoder.predict(val_batch, verbose=0)

    save_batch(train_batch, os.path.join(args.out_dir, "recon_train"), "input")
    save_batch(train_recon, os.path.join(args.out_dir, "recon_train"), "recon")
    save_batch(val_batch, os.path.join(args.out_dir, "recon_val"), "input")
    save_batch(val_recon, os.path.join(args.out_dir, "recon_val"), "recon")

    grid_n = 10
    span = 2.5
    xs = np.linspace(-span, span, grid_n, dtype=np.float32)
    ys = np.linspace(-span, span, grid_n, dtype=np.float32)
    z = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    gen = decoder.predict(z, verbose=0)

    for i in range(gen.shape[0]):
        tf.keras.utils.save_img(os.path.join(args.out_dir, "generated", f"gen_{i:03d}.png"), gen[i])

    h = w = args.img_size
    grid = np.zeros((grid_n * h, grid_n * w, 3), dtype=np.float32)
    k = 0
    for r in range(grid_n):
        for c in range(grid_n):
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w, :] = gen[k]
            k += 1
    tf.keras.utils.save_img(os.path.join(args.out_dir, "generated", "grid.png"), grid)


if __name__ == "__main__":
    main()
