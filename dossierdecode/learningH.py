import numpy as np

def learningH(CaN, Ea) -> "list[float]":
    """
    learning Heff based on the CaN and Ea values with least square method
    Expecting Ea to be a matrix of shape (n, 1) and CaN to be a matrix of shape (n, m)
    """
    ones = np.ones((CaN.shape[0], 1))
    A = np.hstack((ones, CaN))
    # Perform least squares fit
    H = np.linalg.lstsq(A, Ea, rcond=None)[0]
    return H.flatten().tolist()


def meanDifferenceH(CaN, Ea) -> float:
    """
    Calculate the mean difference between the predicted Ea and the actual Ea
    """
    H = learningH(CaN, Ea)
    ones = np.ones((CaN.shape[0], 1))
    A = np.hstack((ones, CaN))
    Ea_predicted = A @ H
    difference = np.mean(np.abs(Ea - Ea_predicted))
    return difference