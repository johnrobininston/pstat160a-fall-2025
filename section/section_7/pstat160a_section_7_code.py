# MC Monte-Carlo

import random, collections
import numpy as np
import matplotlib.pyplot as plt
random.seed(42)

## Problem 1

pi = [0.01,0.39,0.11,0.18,0.26,0.05]

def mh_sample(N=20000, burn=2000):
    x = random.randint(1,6)  # start at state 1 (you can start anywhere)
    samples = []
    for _ in range(N):
        j = random.randint(1,6)            # propose by fair die
        a = min(1, pi[j-1] / pi[x-1])      # acceptance prob
        if random.random() <= a:
            x = j
        samples.append(x)
    return samples[burn:]  # discard burn-in

samps = mh_sample()
counts = collections.Counter(samps)
total = len(samps)
for state in range(1,7):
    print(state, counts[state]/total)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def mh_sample_laplace(N=20000, burn=2000, sigma=1.0, lam=1.0, x0=None, thin=1, rng=None):
    """
    Metropolis-Hastings sampler for Laplace(lambda) using Normal(x, sigma^2) proposals.

    Parameters
    ----------
    N : int
        total iterations (including burn-in)
    burn : int
        number of initial samples to discard
    sigma : float
        proposal standard deviation
    lam : float
        Laplace parameter (density ~ (lam/2) exp(-lam |x|))
    x0 : float or None
        initial state (if None, start at 0)
    thin : int
        thinning factor (keep every `thin`-th sample after burn)
    rng : numpy.random.Generator or None
        random number generator (optional)

    Returns
    -------
    samples : np.ndarray
        array of retained samples (length = (N - burn) // thin)
    accept_rate : float
        overall acceptance rate (including burn-in)
    """
    if rng is None:
        rng = np.random.default_rng()

    if x0 is None:
        x = 0.0
    else:
        x = float(x0)

    samples = []
    n_accept = 0
    for it in range(N):
        # propose y ~ N(x, sigma^2)
        y = rng.normal(loc=x, scale=sigma)

        # log acceptance ratio: log f(y) - log f(x) = -lam * (|y| - |x|)
        log_r = -lam * (abs(y) - abs(x))
        # acceptance probability
        if log_r >= 0:
            a = 1.0
        else:
            a = np.exp(log_r)

        if rng.random() <= a:
            x = y
            n_accept += 1

        # record (we'll apply burn & thinning afterwards)
        samples.append(x)

    accept_rate = n_accept / N
    # discard burn-in and thin
    retained = np.array(samples[burn::thin])
    return retained, accept_rate

# Example usage
if __name__ == "__main__":
    N = 50000
    burn = 5000
    sigma = 1.0     # tune this: smaller sigma -> higher acceptance but slower exploration
    lam = 1.5       # Laplace parameter lambda

    samples, acc = mh_sample_laplace(N=N, burn=burn, sigma=sigma, lam=lam)
    print(f"Accepted fraction: {acc:.3f}")
    print(f"Sample mean ≈ {np.mean(samples):.4f}, sample var ≈ {np.var(samples):.4f}")
    # theoretical mean = 0, var = 2/lambda^2
    print(f"Theoretical var = {2/(lam**2):.4f}")

    # quick diagnostic plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(samples[:1000], marker='.', linestyle='none', markersize=2)
    plt.title("Trace (first 1000 retained samples)")
    plt.xlabel("iteration")
    plt.subplot(1,2,2)
    plt.hist(samples, bins=80, density=True, alpha=0.6)
    # overlay theoretical Laplace density
    xs = np.linspace(-8,8,400)
    laplace_pdf = (lam/2.0) * np.exp(-lam * np.abs(xs))
    plt.plot(xs, laplace_pdf, linewidth=2)
    plt.title("Histogram vs Laplace pdf")
    plt.tight_layout()
    plt.show()


for i in range(10):
    eta = np.exp(1.5*(eta-1))

eta

