import numpy as np

def relu(x):
    return np.maximum(0, x)

x1 = np.array([2.,  1.])
x2 = np.array([0.,  3.])
x3 = np.array([1., -1.])

Wx = np.array([[ 1., -1.],
               [ 0.,  2.]])

Wh = np.array([[ 1.,  0.],
               [-1.,  1.]])

b  = np.array([0., 1.])

h0 = np.array([0., 0.])

h1 = relu(Wx @ x1 + Wh @ h0 + b)

h2 = relu(Wx @ x2 + Wh @ h1 + b)

h3 = relu(Wx @ x3 + Wh @ h2 + b)

h3 = relu(Wx @ x3 + Wh @ h3 + b)


out_sequences = np.array([h1, h2, h3])   # return_sequences=True
out_last = h3                             # return_sequences=False

print("h1 =", h1)
print("h2 =", h2)
print("h3 =", h3)
print("return_sequences:\n", out_sequences)
print("return_last:", out_last)