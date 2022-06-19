import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class max_peak_detector:

    """
    A class representing the max peak detector that uses the time that a variable as its maximum
    compared to the maximums of all other variables in order to detect the leading variable.

    ...

    Attributes
    ----------

    data_window: numpy.array
        A m by t matrix representing a time window of a multivariate signal.

    num_sensors: int
        The number of sensors/variables in the given data window.


    peaks: numpy.array
        A m length array containing the time sample that an individual variable has had its maximum
        in the given data window.

    pulse_location: int
        The m value corresponding to the variable that leads all the others in the given data window.


    Methods
    -------

     get_max_peak():
         Computes the time that each m variable has its maximum value, then uses the relative timings to
         detect the leading variable.

      plot():
          Plots the calculated peaks and the time they occur in the given data window, indicating the leading
          variable.

    """

    def __init__(self, data_window, num_sensors):

        """
        Constructs all the necessary attribute for the lead_lag_detector object.

        Parameters
        ----------

        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        num_sensors: int
            The number of sensors/variables in the given data window.

        """
        self.data_window = data_window.abs().values.T
        self.num_sensors = num_sensors
        self.peaks = np.zeros(num_sensors)
        self.get_max_peak()

    def get_max_peak(self):

        # Find the time of the max absolute peaks for each sensor
        for i, row in enumerate(self.data_window):
            self.peaks[i] = row.argmax()

        # Find the sensor that has its max at the earliest time
        self.pulse_location = self.peaks.argmin()

    def plot(self):

        """
        Plots the calculated peaks and the time they occur in the given data window, indicating the leading
        variable.
        """

        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.size"] = 14
        matplotlib.rcParams["font.family"] = "serif"

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            self.peaks,
            np.arange(self.num_sensors),
            "k.-",
            markersize=8,
            label="Max. peaks",
        )

        ax.text(max(self.peaks) - 10, self.pulse_location + 0.1, "Minimum")
        ax.hlines(
            self.pulse_location,
            min(self.peaks),
            max(self.peaks),
            "r",
            linestyles="dashed",
        )

        ax.set_xlabel("Time: [s]")
        ax.set_yticks(range(self.num_sensors))
        ax.set_yticklabels(["Z%d" % i for i in range(1, self.num_sensors + 1)])

        ax.set_xlim(min(self.peaks) - 3, max(self.peaks) + 3)
        plt.legend()
