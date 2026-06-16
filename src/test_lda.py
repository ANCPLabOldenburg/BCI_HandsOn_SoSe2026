import numpy as np


def test_lda(R, curTest):
    """
    Tests LDA classifier on new data

    Parameters:
    -----------
    R : dict or namedtuple
        Trained LDA model containing:
        - a: weight vector (discriminant coefficients)
        - group1: mean projection for class 1
        - group2: mean projection for class 2
    curTest : numpy.ndarray
        Test data (samples x features)

    Returns:
    --------
    dict
        Contains:
        - prediction: array of predicted class labels (20 or 21)
    """

    # Project test data onto discriminant axis
    y_test = np.dot(R['a'].T, curTest.T)  # Equivalent to R.a' * curTest'

    # Initialize result array with default class
    resTest = np.empty(y_test.shape[1], dtype=int)

    # Calculate distances to each class mean
    dist_to_group1 = np.squeeze(np.abs(y_test.T - R['group1']))
    dist_to_group2 = np.squeeze(np.abs(y_test.T - R['group2']))

    # Assign classes based on minimum distance
    resTest[dist_to_group1 <= dist_to_group2] = 20  # Class 1
    resTest[dist_to_group1 > dist_to_group2] = 21  # Class 2

    return {'prediction': resTest}