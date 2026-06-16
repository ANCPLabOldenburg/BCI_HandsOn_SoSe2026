import numpy as np


def getBalancedTrainset(L, m=None, method='rand'):
    """
    Balances the training set by subsampling classes

    Parameters:
    -----------
    L : numpy.ndarray
        Class labels (non-zero values indicate classes)
    m : int, optional
        Maximum number of samples per class
    method : str, optional
        Subsampling method:
        'rand' - random deselection (default)
        'leading' - deselect from beginning
        'tail' - deselect from end
        'equal' - uniformly spaced deselection

    Returns:
    --------
    labelIdx : numpy.ndarray
        Boolean mask indicating selected samples
    """

    # Get unique classes (ignore zero labels)
    classes = np.setdiff1d(np.unique(L), [0])
    nClasses = len(classes)
    NLabel = np.zeros(nClasses, dtype=int)

    # Find indices for each class
    idx = {}
    for k in range(nClasses):
        idx[k] = np.where(L == classes[k])[0]
        NLabel[k] = len(idx[k])

    # Sort classes by sample count (ascending)
    sort_idx = np.argsort(NLabel)
    n = NLabel[sort_idx]

    # Apply maximum sample limit if specified
    if m is not None and m > 0:
        n[0] = min(n[0], m)

    # Process each class
    for k in range(len(n)):
        class_idx = sort_idx[k]
        current_indices = idx[class_idx]
        current_count = NLabel[class_idx]

        if method.lower() == 'rand':
            # Random deselection
            rand_idx = np.random.permutation(current_count)
            remove_idx = rand_idx[n[0]:]
            L[current_indices[remove_idx]] = 0

        elif method.lower() == 'tail':
            # Deselect from tail
            L[current_indices[n[0]:]] = 0

        elif method.lower() == 'leading':
            # Deselect from beginning
            L[current_indices[:current_count - n[0]]] = 0

        elif method.lower() == 'equal':
            # Uniform deselection
            NRemove = current_count - n[0]
            step_size = current_count / (NRemove + 1)
            remove_idx = np.round(np.arange(step_size, current_count, step_size)).astype(int)
            L[current_indices[remove_idx]] = 0

    # Create boolean mask of selected samples
    labelIdx = L != 0
    return labelIdx