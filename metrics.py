import numpy as np
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy as a fraction of correct predictions.
    """
    correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    return correct_predictions / y_true.shape[0]