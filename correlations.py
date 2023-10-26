import numpy as np

def correlation_2pt(X: np.ndarray, Y: np.ndarray, max_distance = 0):
    n = X.shape[0]
    if X.shape != (n, n) or Y.shape != (n, n):
        raise ValueError('The arguments must be 2-dimensional square arrays')
    if max_distance <= 0:
        max_distance = n // 2 + 1
    return np.array([[d, 0.5 * np.mean(X * np.roll(Y, d, axis=0))
                      + 0.5 * np.mean(X * np.roll(Y, d, axis=1))
                      - np.mean(X) * np.mean(Y)]
                     for d in range(1, max_distance + 1)]).transpose()
