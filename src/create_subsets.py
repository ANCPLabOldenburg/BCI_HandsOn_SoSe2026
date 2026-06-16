import numpy as np


def create_subsets(dat, epoch, ratio=0.2):
    """
    Creates random subsets for feature selection

    Parameters:
    -----------
    dat : numpy.ndarray
        Input data array (typically periodogram data)
    epoch : dict or structured array
        Contains label information with 'label' field
    ratio : float, optional
        Amount of data used for subsets ([0, 1]; default: 0.2 ~ 20%)

    Returns:
    --------
    subSet1 : numpy.ndarray
        Subset of class 1 data
    subSet2 : numpy.ndarray
        Subset of class 2 data
    """

    # Get class labels
    if isinstance(epoch, dict):
        labels = epoch['label']
    else:
        labels = epoch.label

    class1 = np.where(np.array(labels) == 20)[0]
    class2 = np.where(np.array(labels) == 21)[0]

    nb_class1 = len(class1)
    nb_class2 = len(class2)

    # Get random subsets
    subSet1_idx = np.random.choice(class1, size=int(nb_class1 * ratio), replace=False)
    subSet2_idx = np.random.choice(class2, size=int(nb_class2 * ratio), replace=False)

    dat = np.array(dat)
    subSet1 = dat[subSet1_idx, :]
    subSet2 = dat[subSet2_idx, :]

    return subSet1, subSet2