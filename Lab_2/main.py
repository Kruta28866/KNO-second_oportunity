import math

import numpy as np
import tensorflow as tf


def rotate_point_np(point: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Obrót punktu 2D wokół (0,0) z użyciem NumPy.
    point: shape (2,) np.array([x, y])
    angle_rad: kąt w radianach
    returns: shape (2,) np.array([x', y'])
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)  # 2x2
    point_vec = point.astype(np.float32).reshape(2, 1)        # 2x1

    rotated_vec = rotation @ point_vec                        # 2x1
    return rotated_vec.reshape(2,)                            # (2,)


def rotate_point_tf(point: tf.Tensor, angle_rad: tf.Tensor) -> tf.Tensor:
    """
    Obrót punktu 2D wokół (0,0) z użyciem TensorFlow.
    point: shape (2,) lub (2,1)
    angle_rad: skalar (radiany)
    returns: shape (2,)
    """
    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)

    rotation = tf.stack(
        [
            tf.stack([c, -s]),
            tf.stack([s, c]),
        ]
    )  # shape (2,2)

    point_vec = tf.reshape(tf.cast(point, tf.float32), (2, 1))  # shape (2,1)
    rotated_vec = tf.matmul(rotation, point_vec)                # shape (2,1)
    return tf.reshape(rotated_vec, (2,))                         # shape (2,)


def demo() -> None:
    # Przykład: obrót punktu (1, 0) o 90 stopni
    angle = math.pi / 2  # 90° w radianach
    p_np = np.array([1.0, 0.0])

    out_np = rotate_point_np(p_np, angle)
    print("NumPy:", out_np)

    p_tf = tf.constant([1.0, 0.0])
    out_tf = rotate_point_tf(p_tf, tf.constant(angle, dtype=tf.float32))
    print("TensorFlow:", out_tf.numpy())

    # Mini-test (tolerancja, bo floaty mają drobne błędy)
    expected = np.array([0.0, 1.0], dtype=np.float32)
    assert np.allclose(out_np, expected, atol=1e-6), "NumPy test failed"
    assert np.allclose(out_tf.numpy(), expected, atol=1e-6), "TF test failed"
    print("ok")


if __name__ == "__main__":
    demo()
