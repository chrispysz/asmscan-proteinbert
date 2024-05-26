from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import numpy as np
from typing import Tuple


def compute_metrics(y_true: any, y_pred: any) -> Tuple[float, float, np.ndarray[np.float64]]:
    """
        Computes Area Under ROC Score, Average Precision and Rc|FPR metrics.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels.

        y_pred : array-like of shape (n_samples,)
            Target scores.

        Returns
        -------
        roc_auc : float
            Area Under ROC Score.

        pr_auc : float
            Average Precision.

        tpr : np.ndarray[np.float64]
            Interpolation for the TPR parameter based on the number of samples.
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    # Rc|FPR
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Determination of FPR parameter for interpolation of the TPR parameter (based on the number of samples)
    samples = len(y_true)
    order_of_magnitude = int(np.log10(1 / samples))
    fpr_interp = [np.power(10.0, i) for i in range(-1, order_of_magnitude - 1, -1)]

    tpr = np.interp(fpr_interp, fpr, tpr)

    return roc_auc, pr_auc, tpr
