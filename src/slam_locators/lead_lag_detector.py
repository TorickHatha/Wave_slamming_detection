from scipy.ndimage import gaussian_filter1d
from numpy.fft import fft, ifft, fftshift
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class lead_lag_detector:

    """
    A class representing the lead lag detector that uses the lead lag relationships between
    variables in order to determine which variable leads the rest.

    ...

    Attributes
    ----------

    data_window: pandas.DataFrame
        A dataframe representing a m by t matrix of m variables of a multivariate signal.

    num_sensors: int
        The number of sensors/variables in the given data window.

    lead_lag_matrix: numpy.array
        A m by m pairwise matrix representing if a variable lags [0], equal [0.5] or leads [1]
        another variable.

    lead_lag_matrix_un: numpy.array
        A m by m pairwise matrix representing the time sample difference between variables.

    pulse_lags: numpy.array
        A m array that is the rowise mean of the lead_lag_matrix.

    pulse_location: int
        The m value corresponding to the variable that leads all the others in the given data window.


    Methods
    -------
    cross_correlation_using_fft(x, y):
        Computes the cross correlation function between the variables x and y using multiplication
        in the Fourier domain.

     compute_shift(x,y):
         Computes the number of time samples that a variable needs to be shifted to the maximum of the
         cross correlation function.

     get_lead_lag_matrix():
         Computes the lead lag matrix, thresholding it and from this calculating the leading variable.

      plot(matrix_type):
          Plots the calculated lead lag matrix with its associated row wise mean and indicates the leading
          variable.

    """

    def __init__(self, data_window, num_sensors, gaus_param):

        """
        Constructs all the necessary attribute for the lead_lag_detector object

        Parameters
        ----------
        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        num_sensors: int
            The number of sensors/variables in the given data window.

        gaus_param: int
            The parameter used in the gaussian_filter1d() smoothing function.

        """

        if gaus_param == 0:
            self.data_window = data_window.abs().values
        else:
            self.data_window = (
                data_window.abs()
                .apply(lambda x: gaussian_filter1d(x, gaus_param))
                .values
            )

        self.num_sensors = num_sensors
        self.get_lead_lag_matrix()

    def cross_correlation_using_fft(self, x, y):

        f1 = fft(x)
        f2 = fft(np.flipud(y))
        cc = np.real(ifft(f1 * f2))
        return fftshift(cc)

    def compute_shift(self, x, y):

        assert len(x) == len(y)
        c = self.cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        zero_index = int(len(x) / 2) - 1
        shift = zero_index - np.argmax(c)
        return -shift

    def get_lead_lag_matrix(self):

        # initialise all associated variables
        self.lead_lag_matrix = np.zeros((self.num_sensors, self.num_sensors))
        self.lead_lag_matrix_un = np.zeros((self.num_sensors, self.num_sensors))
        self.pulse_lags = np.zeros(self.num_sensors)
        self.pulse_location = 0

        # Compute the pairwise lead lag relationships for the m by m leag lag matrix
        for i in range(self.num_sensors):
            for j in range(self.num_sensors):
                x = self.data_window[:, i]
                y = self.data_window[:, j]

                self.lead_lag_matrix_un[i, j] = self.compute_shift(x, y)

                # Theshold the time sample length to cross-correlation maximum using the
                # heaviside function according to lags [0], equal [0.5] or leads [1]

                self.lead_lag_matrix[i, j] = np.heaviside(self.compute_shift(x, y), 0.5)

        self.pulse_lags = self.lead_lag_matrix.mean(axis=1)
        self.pulse_location = self.lead_lag_matrix.mean(axis=1).argmin()

    def plot(self, matrix_type="thesholded"):

        """
        Plots the calculated lead lag matrix with its associated row wise mean and indicates the leading
        variable.

        Parameters
        ----------
        matrix_type: str

            The matrix type to use, can be 'thesholded'- self.lead_lag_matrix
            or 'unthresholded' - self.lead_lag_matrix_un.

        Returns
        -------

        A plot of the chosen lead lag matrix with the associated row wise mean and leading variable
        indicator.

        """

        if matrix_type == "thresholded":
            chosen_matrix = self.lead_lag_matrix
        if matrix_type == "unthresholded":
            chosen_matrix = self.lead_lag_matrix_un

        matplotlib.rcParams.update({"font.size": 12})

        fig3 = plt.figure(constrained_layout=True, figsize=(10, 6))
        gs = fig3.add_gridspec(3, 3)

        # Pulse lag plot
        fig3_ax1 = fig3.add_subplot(gs[:, 0])

        fig3_ax1.plot(self.pulse_lags, range(self.num_sensors), "k.-")

        fig3_ax1.set_yticks(range(self.num_sensors))
        fig3_ax1.set_ylim(-0.5, self.num_sensors - 0.5)
        fig3_ax1.set_yticklabels(["Z%d" % i for i in range(1, self.num_sensors + 1)])

        fig3_ax1.hlines(
            self.pulse_location, 0, max(self.pulse_lags), "r", linestyles="dashed"
        )
        fig3_ax1.set_title(r"$\mu_A$")
        fig3_ax1.text(max(self.pulse_lags) - 0.2, self.pulse_location + 0.1, "Minimum")

        # Lead_lag matrix plot
        fig3_ax2 = fig3.add_subplot(gs[:, 1:3])

        im = fig3_ax2.imshow(
            chosen_matrix, interpolation="nearest", cmap="gray", origin="lower"
        )
        fig3_ax2.yaxis.tick_right()
        fig3_ax2.xaxis.tick_bottom()
        fig3_ax2.set_yticks(range(self.num_sensors))
        fig3_ax2.set_yticklabels(["Z%d" % i for i in range(1, self.num_sensors + 1)])
        fig3_ax2.set_xticks(range(self.num_sensors))
        fig3_ax2.set_xticklabels(["Z%d" % i for i in range(1, self.num_sensors + 1)])
        fig3_ax2.set_title("Lead-lag matrix A")
        fig3.colorbar(im)
