import numpy as np
from scipy.linalg import pinv


def train_lda(curTrain, curOBJECTIVE):
    """
    Trains a Linear Discriminant Analysis (LDA) classifier

    Parameters:
    -----------
    curTrain : numpy.ndarray (m x n)
        Training data (m samples, n features)
    curOBJECTIVE : numpy.ndarray
        Class labels (must contain exactly 2 classes)

    Returns:
    --------
    dict
        Contains LDA model parameters:
        - 'a': discriminant weights (n x 1)
        - 'group1': projection of class 1 mean
        - 'group2': projection of class 2 mean
    """

    # Get unique class labels (should be exactly 2)
    classLabel = np.unique(curOBJECTIVE)
    if len(classLabel) != 2:
        raise ValueError("LDA requires exactly 2 classes")

    m, n = curTrain.shape

    # Initialize
    groupMeans = np.zeros((2, n))
    s_w = np.zeros((n, n))  # Within-class scatter matrix

    # Compute statistics for each class
    for i in range(2):
        # Get samples for current class
        group = np.where(curOBJECTIVE == classLabel[i])[0]

        # Compute class mean
        groupMeans[i, :] = np.mean(curTrain[group, :], axis=0)

        # Compute covariance matrix (with ddof=1 for unbiased estimate)
        class_cov = np.cov(curTrain[group, :], rowvar=False, ddof=1)
        s_w += (len(group) - 1) * class_cov

    # Compute discriminant weights (using pseudo-inverse for stability)
    mean_diff = (groupMeans[0, :] - groupMeans[1, :]).reshape(-1, 1)

    # w \proto S_w^{-1} (\bar{x}_{c1} - \bar{x}_{c2})
    # S_w w = \bar{x}_{c1} - \bar{x}_{c2}
    # np.linalg.solve for solving Ax=b, but A must be full rank matrix
    # equivalent to MATLAB backslash \ operator (which supports any shape of A, by using Least-square)
    # for Symmetric Positive Definite Matrix: np.linalg.cholesky (more easier solution)
    # S_w must be symmetric, but half-positive definite, as features may more than samples (most of ML case)

    # wrong calculation (maybe, check operators and dimensions)
    # May due to 'inverse'

    # R = {
    #     'a': pinv(s_w) @ mean_diff,  # Equivalent to s_w \ mean_diff in MATLAB
    #     'group1': (groupMeans[0, :] @ pinv(s_w) @ groupMeans[0, :].T),
    #     'group2': (groupMeans[1, :] @ pinv(s_w) @ groupMeans[1, :].T)
    # }

    # Using np.linalg.solve
    a = np.linalg.solve(s_w, mean_diff)
    R = {
        'a': a,
        'group1': np.dot(groupMeans[0, :], a),
        'group2': np.dot(groupMeans[1, :], a)
    }

    return R