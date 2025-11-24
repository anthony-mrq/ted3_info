from sklearn.metrics import root_mean_squared_error
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error between true and predicted values.

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
    Returns:
        float: Root Mean Squared Error value
    """
    return root_mean_squared_error(y_true, y_pred)
