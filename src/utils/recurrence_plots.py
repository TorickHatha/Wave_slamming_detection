from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np


def calculate_recurrence_plot(data_window: np.array, method: str) -> np.array:

    """
    Given a matrix containing data of multivariate signal, create a recurrence plot according
    to a specific distance metric.

    param data_window: An m by t matrix of m variables of a multivariate signal.
    param method: The distance metric to use. The distance function can be 'euclidean','cosine','mahalanobis'
        or any of the methods implemented in the scipy.spatial.distance package.

    return : A m by m matrix representing the recurrence plot of the given data window.
    """

    return squareform(pdist(data_window, method))
