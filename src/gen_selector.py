import numpy as np

def gen_selector(n_samples, n_folds, random_seed=None):
    """
    Generate a random selector for cross-validation

    Parameters:
        n_samples: sample numbers
        n_folds: cross validation folds
        random_seed: optional, can reproduce results

    return:
        selector: (n_samples)
    """

    selector = np.ceil((np.arange(1, n_samples + 1)) / (n_samples / n_folds))
    selector = selector[np.random.default_rng(seed = random_seed).permutation(n_samples)]

    return selector.astype(int)
