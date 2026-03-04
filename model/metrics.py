import numpy as np


def accuracy_score_(y, y_pred):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """

    correct_classification = 0
    for itrue, ipred in zip(y, y_pred): 
        if itrue == ipred:
            correct_classification += 1
    return correct_classification / len(y)


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        his function should not raise any Exception.
    """


