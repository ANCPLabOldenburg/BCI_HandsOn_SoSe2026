import numpy as np
from sklearn.svm import SVC
from ecogGetDefC import ecogGetDefC

"""
Update Log:
25/05/2025 (XW):    - decision_values in dict `svm_results` added, for further ROC plot
                    - parameter added: dispC, False default
"""

def classification_svm(dat, realClassLabels, selector, N, optimizeC=False, dispC=False):
    """
    SVM classification with optional C parameter optimization

    Parameters:
        dat: Data matrix (n_samples x n_features)
        realClassLabels: True class labels (n_samples,)
        selector: CV partitioning indices
        N: Number of CV folds
        optimize_C: Whether to optimize C parameter (default: False)
        dispC: whether print the current using C value (default: False)

    Returns:
        results: Dictionary containing:
            - accuracy: Overall classification accuracy across all folds
            - accuracies: Accuracy for each individual fold
            - best_Cs: Best C values for each fold
            - weights: SVM weights for each fold
            - biases: SVM biases for each fold
            - predictedClassLabels: Predicted labels (n_samples,)
            - decision_values: Decision function values (n_samples,)
    """

    svm_results = {
        'accuracy': [],
        'accuracies': np.zeros(N),
        'best_Cs': np.zeros(N),
        'weights': np.zeros((N, dat.shape[1])),
        'biases': np.zeros(N),
        'predictedClassLabels': np.zeros_like(realClassLabels),
        'decision_values': np.zeros_like(realClassLabels, dtype=float),
        'C_plus_W': []
    }

    for k in range(1, N + 1):
        print(f'CV step #{k}')

        # Split data into train/test sets
        testIdx = np.where(selector == k)[0]
        trainIdx = np.setdiff1d(np.arange(len(realClassLabels)), testIdx)

        X_train = dat[trainIdx, :]
        y_train = realClassLabels[trainIdx]
        X_test = dat[testIdx, :]
        y_test = realClassLabels[testIdx]

        # ------ C Parameter Optimization ------
        defaultC = ecogGetDefC(X_train)  # Get default C using Joachims' method

        if optimizeC:  # with parameter optimization

            # Generate C search range (geometric progression around default_C)
            c_h = [defaultC + (1 / 3) * defaultC]
            c_l = [defaultC - (1 / 3) * defaultC]
            for i in range(1, 15):
                c_h.append(c_h[i - 1] + (1 / 3) * c_h[i - 1])
                c_l.append(c_l[i - 1] - (1 / 3) * c_l[i - 1])

            # Combine search range: [descending C_low] + [default_C] + [ascending C_high]
            iterative_c = c_l[::-1] + [defaultC] + c_h

            # Alternative:
            # iterative_c = np.logspace(np.log10(defaultC) - 2, np.log10(defaultC) + 2, 31)

            optAcc = []

            # Grid search for optimal C
            for m, C in enumerate(iterative_c):
                svm = SVC(C=C, kernel='linear')
                svm.fit(X_train, y_train)

                # Evaluate
                pred = svm.predict(X_test)
                accuracy = np.mean(pred == y_test)
                optAcc.append(accuracy)

                # print(f"C={C:.2f}, Accuracy={accuracy:.2f}, ||w||={np.linalg.norm(svm.coef_):.2f}")

            # Select best C
            best_idx = np.argmax(optAcc)
            best_C = iterative_c[best_idx]

            if dispC: print(f"Selected C: {best_C}")
            svm_results['best_Cs'][k - 1] = best_C

            # Retrain with best C
            svm = SVC(C=best_C, kernel='linear')
            svm.fit(X_train, y_train)

        else:
            # Without C-optimization
            print(f"Using default C: {defaultC}")

            # Train
            svm = SVC(C=defaultC, kernel='linear')
            svm.fit(X_train, y_train)

        # Get Results from SVM using best C
        svm_results['predictedClassLabels'][testIdx] = svm.predict(X_test)
        svm_results['decision_values'][testIdx] = svm.decision_function(X_test)
        svm_results['accuracies'][k - 1] = np.mean(svm.predict(X_test) == y_test)
        svm_results['weights'][k - 1, :] = svm.coef_[0]
        svm_results['biases'][k - 1] = svm.intercept_[0]
        # C_plus_W: Tuple of (C_value, norm_of_weights)
        svm_results['C_plus_W'].append((svm.C, np.linalg.norm(svm.coef_)))

    svm_results['accuracy'] = np.mean(svm_results['predictedClassLabels'] == realClassLabels)

    return svm_results
