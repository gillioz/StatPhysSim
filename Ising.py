import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import trange

def Ising_MC(n: int, beta: float, fig_number = 1, steps = 100, sleep = 0):
    # initialize the system in a random state
    sigma = np.random.choice([-1, 1], size=(n, n))
    epsilon = -2 * sigma * (
        np.roll(sigma, 1, axis=0) + np.roll(sigma, -1, axis=0)
        + np.roll(sigma, 1, axis = 1) + np.roll(sigma, -1, axis = 1)
    )
    if fig_number > 0:
        fig = plt.figure(fig_number)
        image = plt.imshow(sigma)
        plt.show()
    # let the system evolve at temperature T = 1 / beta
    for t in trange(steps):
        for k in range(n * n):
            # pick a site at random
            i, j = np.random.randint(n, size=2)
            # compute the energy difference dE in case of a spin flip
            dE = -epsilon[i, j]
            # flip sigma if dE is negative, or otherwise with probability e^(-beta * dE)
            if dE <= 0 or np.exp(-beta * dE) > np.random.rand():
                sigma[i, j] *= -1
                s = sigma[i, j]
                epsilon[i, j] *= -1
                epsilon[(i + 1) % n, j] -= 4 * s * sigma[(i + 1) % n, j]
                epsilon[(i - 1) % n, j] -= 4 * s * sigma[(i - 1) % n, j]
                epsilon[i, (j + 1) % n] -= 4 * s * sigma[i, (j + 1) % n]
                epsilon[i, (j - 1) % n] -= 4 * s * sigma[i, (j - 1) % n]
        if fig_number > 0:
            image.set_data(sigma)
            fig.canvas.draw()
        if sleep > 0:
            time.sleep(sleep)
    return sigma, epsilon

# critical temperature
Ising_beta_c = np.log(1 + np.sqrt(2)) / 2
