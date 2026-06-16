import numpy as np
from collections import namedtuple


def test_bayes(R, curTest):
    """
    Tests a naive Bayes classifier on new data

    Parameters:
    -----------
    R : list of namedtuples
        Trained model from train_bayes()
    curTest : numpy.ndarray
        Test data (samples x features)

    Returns:
    --------
    dict with prediction results
    """

    # Input validation
    if len(R[0].meanC) != curTest.shape[1]:
        raise ValueError('Number of attributes in R and TEST must be equal.')

    nclasses = len(R)
    nbt = [{'pC': np.zeros_like(curTest), 'postsC': None} for _ in range(nclasses)]

    # Calculate probability densities for each attribute
    for c in range(nclasses):
        sC = 2 * R[c].varC
        dC = curTest - R[c].meanC
        nbt[c]['pC'] = np.exp(-dC ** 2 / sC) / np.sqrt(np.pi * sC)
        nbt[c]['postsC'] = nbt[c]['pC'][:, 0].copy()

    # Calculate posterior probabilities
    for k in range(1, curTest.shape[1]):
        postsSum = np.zeros(curTest.shape[0])
        for c in range(nclasses):
            nbt[c]['postsC'] *= nbt[c]['pC'][:, k]
            postsSum += nbt[c]['postsC']

        # Handle overflow/underflow
        OUflow = np.where((postsSum < 1e-24) | (postsSum > 1e24))[0]
        if len(OUflow) > 0:
            for c in range(nclasses):
                nbt[c]['postsC'][OUflow] /= postsSum[OUflow]

    # Normalize probabilities
    postsSum = np.zeros(curTest.shape[0])
    for c in range(nclasses):
        postsSum += nbt[c]['postsC']
    postsNorm = 1 / postsSum

    for c in range(nclasses):
        nbt[c]['postsC'] *= postsNorm

    # Make predictions
    Res = {'prediction': np.full(curTest.shape[0], R[0].label)}
    postsC = nbt[0]['postsC'].copy()

    for c in range(1, nclasses):
        greater = nbt[c]['postsC'] > postsC
        postsC[greater] = nbt[c]['postsC'][greater]
        Res['prediction'][greater] = R[c].label

    return Res