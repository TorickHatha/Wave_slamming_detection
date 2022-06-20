import src.slam_locators.max_peak_detector as sut
import numpy as np
import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_array_equal

ARRAY_1 = np.zeros(100)
ARRAY_1[50] = 1
ARRAY_2 = np.zeros(100)
ARRAY_2[45] = 1


def test_calculate_max_peaks_and_location():

    data_window = np.array([ARRAY_1, ARRAY_2, ARRAY_1])
    max_peak_times, pulse_location = sut.calculate_max_peaks_and_location(data_window)

    assert_array_equal(max_peak_times, [50, 45, 50])
    assert pulse_location == 1


def test_plot_max_peaks_and_location():

    data_window = np.array([ARRAY_1, ARRAY_2, ARRAY_1])
    max_peak_times, pulse_location = sut.calculate_max_peaks_and_location(data_window)
    fig = sut.plot_max_peaks_and_location(max_peak_times, pulse_location)
    file_path = "tests/tests_output/"
    plt.savefig(file_path + "max_peak_detector.png")
