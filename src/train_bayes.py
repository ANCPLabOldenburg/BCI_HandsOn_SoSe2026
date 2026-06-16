import numpy as np
from collections import namedtuple


def train_bayes(curTrain, curOBJECTIVE):
    """
    Trains a naive Bayes classifier

    Parameters:
    -----------
    curTrain : numpy.ndarray
        Training data (samples x features)
    curOBJECTIVE : numpy.ndarray
        Class labels for training data

    Returns:
    --------
    list of trained class models
    """

    EPSILON = 5e-13

    # Create a namedtuple to store class statistics (similar to MATLAB struct)
    ClassStats = namedtuple('ClassStats', ['label', 'meanC', 'varC'])
    R = []

    classes = np.unique(curOBJECTIVE)

    for k, class_label in enumerate(classes):
        # Get indices for current class
        idx = np.where(curOBJECTIVE == class_label)[0]

        # Calculate mean and variance
        meanC = np.mean(curTrain[idx, :], axis=0)
        varC = np.var(curTrain[idx, :], axis=0, ddof=1)  # ddof=1 for sample variance

        # Apply variance flooring
        varC[varC < EPSILON] = EPSILON

        # Store results (using namedtuple instead of MATLAB struct)
        R.append(ClassStats(label=class_label, meanC=meanC, varC=varC))

    return R