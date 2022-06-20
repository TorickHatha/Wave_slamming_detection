import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Tuple


def calculate_max_peaks_and_location(data_window: np.array) -> Tuple[np.array, int]:
    """
    Computes the time that each m variable has its maximum value, then uses the relative timings to
    detect the leading variable.

    :param data_window: m x n matrix of m variables.
    :return: The timings of the max peaks and the pulse location.
    """
    max_peak_times = []
    # Find the time of the max absolute peaks for each sensor
    for sensor in abs(data_window):
        max_peak_times.append(sensor.argmax())

    # Find the sensor that has its max at the earliest time
    max_peak_times = np.array(max_peak_times)
    pulse_location = max_peak_times.argmin()

    return max_peak_times, pulse_location


def plot_max_peaks_and_location(
    max_peak_times: np.array, pulse_location: int
) -> plt.figure:
    """
    Plots the calculated peaks and the time they occur in the given data window, indicating the leading
    variable.
    """
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 6))
    num_sensors = max_peak_times.shape[0]
    ax.plot(
        max_peak_times,
        np.arange(num_sensors),
        "k.-",
        markersize=8,
        label="Max. peaks",
    )

    ax.text(max(max_peak_times) * 0.97, pulse_location + 0.1, "Minimum")
    ax.hlines(
        pulse_location,
        min(max_peak_times),
        max(max_peak_times),
        "r",
        linestyles="dashed",
    )

    ax.set_xlabel("Time: [s]")
    ax.set_yticks(range(num_sensors))
    tick_labels = [f"sensor {i}" for i in range(1, num_sensors + 1)]
    ax.set_yticklabels(tick_labels)

    ax.set_xlim(min(max_peak_times) - 3, max(max_peak_times) + 3)
    plt.legend()

    return fig
