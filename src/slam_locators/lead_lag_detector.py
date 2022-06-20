from scipy.ndimage import gaussian_filter1d
from numpy.fft import fft, ifft, fftshift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple


def calculate_lead_lag_matrix(
    data_window: np.array, threshold_matrix: bool = True
) -> np.array:
    """
    Describes the lead lag relationship between variables.

    :param data_window: m x n matrix of m variables.
    :param threshold_matrix: Thresholding flag.
    :return: m x m , lead lag matrix.
    """

    lead_lag_matrix = np.zeros((data_window.shape[0], data_window.shape[0]))
    for i, x in enumerate(data_window):
        for j, y in enumerate(data_window):

            # Theshold the time sample length to cross-correlation maximum using the
            # heaviside function according to lags [0], equal [0.5] or leads [1]
            if i == j:
                lead_lag_matrix[i, j] = 0.5
            else:
                shift = _compute_shift(x, y)
                lead_lag_matrix[i, j] = (
                    np.heaviside(shift, 0.5) if threshold_matrix else shift
                )

    return lead_lag_matrix


def _compute_shift(x: np.array, y: np.array) -> int:
    """
    Computes the relative time shift between two arrays.

    :param x: Array 1.
    :param y: Array 2.
    :return: Number of units x leads y.
    """

    if len(x) != len(y):
        raise ValueError(f"Input arrays not of equal length: x-{len(x)} vs y-{len(y)}")

    cross_correlation = _compute_cross_correlation_fft(x, y)

    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(cross_correlation)
    return -shift


def _compute_cross_correlation_fft(x: np.array, y: np.array) -> np.array:
    """
    Computes the cross correlation between two arrays through
    multiplcation in the fourier domain.

    :param x: Array 1.
    :param y: Array 2.
    :return: cross correlation array.
    """
    # calculate the fft of each array
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    # inverse transform the convolution of
    # array 1 with array 2
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)


def get_pulse_lags_and_location(lead_lag_matrix: np.array) -> Tuple[np.array, int]:
    """
    :param lead_lag_matrix: m x m matrix containg the pairwise lead lag relationship between variables.
    :return: _description_
    """
    pulse_lags = lead_lag_matrix.mean(axis=1)
    pulse_location = lead_lag_matrix.mean(axis=1).argmin()
    return pulse_lags, pulse_location


def plot_lead_lag_matrix(lead_lag_matrix: np.array) -> plt.figure:
    """
    Plots the calculated lead lag matrix with its associated row wise mean
    and indicates the leading variable.

    :param lead_lag_matrix: m x m matrix containg the pairwise lead lag relationship between variables.

    :return :A plot of the chosen lead lag matrix with the associated row wise mean
    and leading variable indicator.
    """

    matplotlib.rcParams.update({"font.size": 12})
    num_sensors = lead_lag_matrix.shape[0]
    pulse_lags, pulse_location = get_pulse_lags_and_location(lead_lag_matrix)
    plt.cla()
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = fig.add_gridspec(3, 3)

    # Pulse lag plot
    fig_ax1 = fig.add_subplot(gs[:, 0])

    fig_ax1.plot(pulse_lags, range(num_sensors), "k.-")

    fig_ax1.set_yticks(range(num_sensors))
    fig_ax1.set_ylim(-0.5, num_sensors - 0.5)
    tick_labels = [f"sensor {i}" for i in range(1, num_sensors + 1)]
    fig_ax1.set_yticklabels(tick_labels)

    fig_ax1.hlines(pulse_location, 0, max(pulse_lags), "r", linestyles="dashed")
    fig_ax1.set_title(r"$\mu_A$")
    fig_ax1.text(max(pulse_lags) - 0.2, pulse_location + 0.1, "Minimum")

    # Lead_lag matrix plot
    fig_ax2 = fig.add_subplot(gs[:, 1:3])

    im = fig_ax2.imshow(
        lead_lag_matrix, interpolation="nearest", cmap="gray", origin="lower"
    )
    fig_ax2.yaxis.tick_right()
    fig_ax2.xaxis.tick_bottom()
    fig_ax2.set_yticks(range(num_sensors))
    fig_ax2.set_yticklabels(tick_labels)
    fig_ax2.set_xticks(range(num_sensors))
    fig_ax2.set_xticklabels(tick_labels)
    fig_ax2.set_title("Lead-lag matrix")
    fig.colorbar(im)
    return fig
