import src.slam_locators.lead_lag_detector as sut
import numpy as np
import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_array_equal

ARRAY_1 = np.zeros(100)
ARRAY_1[50:60] = 1
ARRAY_2 = np.zeros(100)
ARRAY_2[40:50] = 1


def test_compute_shift():

    shift = sut._compute_shift(ARRAY_1, ARRAY_2)

    assert shift == 10


def test_calculate_lead_lag_matrix():

    lead_lag_matrix = sut.calculate_lead_lag_matrix(np.array([ARRAY_1, ARRAY_2]))

    assert_array_equal(lead_lag_matrix, np.array([[0.5, 1.0], [0.0, 0.5]]))


def test_plot_lead_lag_matrix():

    lead_lag_matrix = sut.calculate_lead_lag_matrix(
        np.array([ARRAY_1, ARRAY_2, ARRAY_1])
    )
    fig = sut.plot_lead_lag_matrix(lead_lag_matrix)
    file_path = "tests/slam_locators/test_output/"
    plt.savefig(file_path + "lead_lag_matrix.png")
