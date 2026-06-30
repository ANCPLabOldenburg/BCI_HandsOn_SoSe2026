import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_ROC(svm_results, test_labels, do_plot=True):
    """
    Plot ROC curve directly from SVM results dictionary

    Parameters
    ----------
    svm_results : dict
        SVM results containing:
        - decision_values : ndarray (n_samples,)
        - weights : ndarray (n_folds, n_features)
    test_labels : ndarray (n_samples,)
        True binary labels
    do_plot : bool
        Whether to display the plot

    Returns
    -------
    auc : float
        Area Under the ROC Curve
    fig : matplotlib Figure or None
        Only returned if do_plot=True
    """
    # Get decision values and weights based on fold selection
    best_cv_idx = np.argmax(svm_results['accuracies'])

    y_score = svm_results['decision_values']
    y_true = test_labels
    W = np.array(svm_results['weights'][best_cv_idx, :])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=21)
    roc_auc = auc(fpr, tpr)

    if do_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Distance distribution
        distances = y_score / np.linalg.norm(W)
        ax1.hist(distances[y_true == 20], bins=30, alpha=0.5, color='b', label='Class 20')
        ax1.hist(distances[y_true == 21], bins=30, alpha=0.5, color='r', label='Class 21')
        ax1.set_xlabel('Normalized Distance to Hyperplane')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.set_title(f'Class Separation in CV {best_cv_idx}')

        # Panel 2: ROC curve
        ax2.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate (Class 20)')
        ax2.set_ylabel('True Positive Rate (Class 21)')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.show()
        return roc_auc, fig

    return roc_auc