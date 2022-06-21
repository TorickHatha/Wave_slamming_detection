from skimage.transform import resize
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Flatten
from pyts import utils
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap as umap_decomposition
from sklearn.cluster import DBSCAN
from src.utils.recurrence_plots import calculate_recurrence_plot


class wave_slam_detector:

    """
    A class representing the wave slam detector that splits a given multivariate signal into discrete time
    windows, converts each window into an image either a type of Recurrence plot or simply just the time
    window. Features are then extracted from the converted images. The resulting high dimensional feature vectors
    are reduced in dimension using PCA and UMAP. The time windows are clustered in the reduced dimensional space.
    The found clusters are mapped back to the time domain and the associated label of interested is chosen. This
    chosen label is clustered in time quantifying the instances of the desired events.
    ...

    Attributes
    ----------

    df: pandas.DataFrame
        A dataframe representing a m by t matrix of m variables of a multivariate signal.

    windowed_index: numpy.array
        A w length array containing the indices for each time window.

    feature_vectors: numpy.array
        A w by f matrix where f is the number of extracted features and w is the number of time windows.

    pca_model: sklearn.decomposition._pca.PCA
        The PCA model created for the associated max_components number of chosen principle components.

    pca_transformed: numpy.array
        A w by p matrix where p is the number of principle components used in the pca_model and w is the
        number of time windows.

    umap_model: umap.umap_.UMAP
        The UMAP model created for the associated n_neighbors and min_dist parameters used.

    umap_transformed: numpy.array
        A w by 2 matrix where w is the number of time windows, representing the time windows in their features
        in the UMAP reduced two dimensional space.

    DBSCAN_feature_labels: numpy.array
        A w length array where w is the number if time windows, of the found DBSCAN cluster labels in the UMAP
        reduced space.

     chosen_cluster: int
         After user evaluation this the cluster associated with the phenomenon of interest.

     DBSCAN_time_labels: numpy.array
         A w length array where w is the number if time windows, of the found DBSCAN cluster labels of the chosen
         cluster in the time domain.

     results_df: pandas.DataFrame
         A dataframe containing the information relating to the found impacts. It contains the number of the impact found,
         the time sample index of the start and ending time window the impact occurred in and the approximate time sample
         within this time range that the impact occured.

    Methods
    -------

    feature_extraction(window_size,stride,compressed_window_size,image_type):

        Splits the given multivariate signal into time windows according to the given window_size and stride parameters.
        The time windows are converted to an image according to the image_type string. The images are compressed and
        passed through the pretrained VGG16 CNN. The resulting feature vectors per image are stored in a matrix.

    PCA(max_components):

        Decomposes the feature vectors found through the CNN to max_components number of principle components.

    UMAP(n_neighbors,min_dist):

        Decomposes the PCA reduced feature vectors found into a two dimensionsal space using the UMAP algorithm.

     clustering(eps,min_samples):

         Clusters the time windows represented in the UMAP feature space using the DBSCAN algorithm.

     user_cluster_evaluation():

         Plots the found cluster labels corresponding to the time domain of the original signal, allowing the user
         to compare the cluster labels to the original signal.

     clustering_in_time():

         Clusters the chosen cluster label in time using the DBSCAN algorithm. Extracts the time at which each cluster
         in time starts and ends. The approximate impact time within this time range is also calculated. These values
         are then stored in a pandas.DataFrame.

    """

    def __init__(self, dataframe):

        """
        Constructs all the necessary attribute for the wave_slam_detector object.

        Parameters
        ----------

        data_window: pandas.DataFrame
            A dataframe representing a m by t matrix of m variables of a multivariate signal.

        """
        self.df = dataframe

    def feature_extraction(
        self, window_size, stride, compressed_window_size, image_type
    ):

        """
        Given a multivariate signal, segment it into discrete time windows of a specfied size and
        at a set stride. Convert these time windows to images using the recurrence_plot() or use
        the given multivariate data window as an image. Pass these images through a the pre-
        trained VGG16 CNN, extracting the final max-pooling layer.

        Parameters
        ----------
        window_size: int
            The number of sample in each time window.

        stride: int
            The number of samples between each time window.

        compressed_window_size: int
            The size of the image (number of pixels n by n) passed to the CNN.

        image_type:
            The type of image the time window should be converted to. If the recurrence plot
            representation is used the 'euclidean','cosine' or 'mahalanobis' method is passed. If
            the time window is to be used as an image then 'none' is passed.

        """

        # Create array of indices for each time window
        self.windowed_index = utils.windowed_view(
            np.arange(0, self.df.shape[0]).reshape(1, -1), window_size, stride
        )[0]

        # set the input image size for the CNN
        if compressed_window_size == 0:
            cnn_window_size = window_size
        else:
            cnn_window_size = compressed_window_size

        # load pretrained CNN model
        model = VGG16(
            include_top=False, input_shape=(cnn_window_size, cnn_window_size, 3)
        )
        flat1 = Flatten()(model.layers[-1].output)
        model = Model(inputs=model.inputs, outputs=flat1)

        # initialise the feature vector
        VGG16_feature_vectors = []

        # Loop through time windows
        for window in self.windowed_index:

            # Raw data window
            img_data = self.df.iloc[window, :]

            # Generate image representing the time window
            if image_type == "none":
                img = resize(
                    img_data.abs().values,
                    (cnn_window_size, cnn_window_size),
                    anti_aliasing=True,
                )
            else:
                img = recurrence_plot(img_data, image_type)

            # Compress time window
            if compressed_window_size != 0:
                img = resize(
                    img,
                    (compressed_window_size, compressed_window_size),
                    anti_aliasing=True,
                )

            # Generate image tensor
            gg = np.zeros((img.shape[0], img.shape[1], 3))
            gg[0 : img.shape[0], 0 : img.shape[1], 0] = img
            gg[0 : img.shape[0], 0 : img.shape[1], 1] = img
            gg[0 : img.shape[0], 0 : img.shape[1], 2] = img

            x = np.expand_dims(gg, axis=0)
            # Apply VGG16 image preprocessing
            x = preprocess_input(x)
            # extract feature vector
            img_feature_vector = model.predict(x)[0]
            VGG16_feature_vectors.append(img_feature_vector)

        # Create feature vector attribute
        self.feature_vectors = VGG16_feature_vectors

    def PCA(self, max_components, plot_variance=False):

        """
        Decomposes the feature vectors found through the CNN to max_components number of principle components.

        Parameters
        ----------
        max_components: int
            The number of principle components used in the PCA model.

        plot_variance: bool
            States if the cumulative variance plot for the model should be visualised.

        """

        # Initialise PCA model
        self.pca_model = scikit_PCA(n_components=max_components)

        # Fit and transform the feature vectors to the PCA domain
        self.pca_transformed = self.pca_model.fit_transform(self.feature_vectors)

        # Plot the cumulative variance
        if plot_variance:

            fig, ax = plt.subplots(figsize=(10, 5))
            plt.plot(
                np.cumsum(self.pca_model.explained_variance_ratio_), marker=".", c="k"
            )
            plt.xticks(range(max_components))
            ax.set_xticklabels(np.arange(max_components) + 1)
            plt.title("PCA Components cumulative  variance")
            plt.xlabel("No. of components")
            plt.ylabel("Variance")

    def UMAP(self, n_neighbors, min_dist, plot=False, connect_plot=False):

        """
        Projects the PCA reduced domain feature vectors to two dimensions using the UMAP algorithm.

        Parameters
        ----------
        n_neighbors: int
            The number of neighbours connected in the UMAP algorithm.

        min_dist: int
            The minimum distance between points in the reduced feature space.

        plot: bool
            States if the UMAP reduced feature space should be visualised.

        connect_plot: bool
            States if the UMAP reduced feature space should be visualised where each point is connected
            to the next point in time.
        """
        # Initialise the UMAP model
        self.umap_model = umap_decomposition.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=18, n_components=2
        )

        # Fit and transform the PCA reduced feature vectors using the UMAP algorithm
        self.umap_transformed = self.umap_model.fit_transform(self.pca_transformed)

        # Plot the UMAP reduced feature space
        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                self.umap_transformed[:, 0],
                self.umap_transformed[:, 1],
                c="k",
                marker=".",
            )
            ax.set_title("UMAP decomposition")
        if connect_plot:
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.plot(
                self.umap_transformed[:, 0],
                self.umap_transformed[:, 1],
                "k.-",
                alpha=0.5,
            )
            ax.set_title("UMAP decomposition")

    def clustering(self, eps, min_samples, plot=False):

        """
        Clusters the time windows represented in the UMAP feature space using the DBSCAN algorithm.

        Parameters
        ----------

        eps: int
            The radius, size of neighbourhood around each point in the DBSCAN algorithm.

        min_samples:
            The minimum number of samples in a neighbourhood to be considered a 'core' point.

        plot: bool
            States if the UMAP reduced feature space should be visualised, coloured with the found
            DBSCAN cluster labels.

        """

        # Initialise the DBSCAN algorithm
        DBSCAN_model = DBSCAN(eps=eps, min_samples=min_samples).fit(
            self.umap_transformed
        )

        # Create the feature space label attribute
        self.DBSCAN_feature_labels = DBSCAN_model.labels_ + 1

        # Plot the UMAP feature space with the found DBSCAN cluster labels
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("UMAP decomposition")

            for col in np.unique(self.DBSCAN_feature_labels):
                ax.scatter(
                    self.umap_transformed[self.DBSCAN_feature_labels == col, 0],
                    self.umap_transformed[self.DBSCAN_feature_labels == col, 1],
                    marker=".",
                    label="Cluster %d" % col,
                )

            ax.set_xlabel(r"$U_1$")
            ax.set_ylabel(r"$U_2$")

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            plt.tight_layout()

    def user_cluster_evaluation(self):

        """
        Plots the found cluster labels corresponding to the time domain of the original signal, allowing the user
        to compare the cluster labels to the original signal.

        """

        # Create empty chosen cluster attribute
        self.chosen_cluster = []

        fig, ax = plt.subplots(2, 1, figsize=(15, 6))

        # Plot DBSCAN labels
        ax[0].plot(self.DBSCAN_feature_labels, "k.", markersize=10)
        ax[0].set_ylabel("Cluster label")
        ax[0].xaxis.set_visible(False)
        ax[0].set_title("DBSCAN cluster labels")
        ax[0].set_ylim(0, max(self.DBSCAN_feature_labels) + 1)
        ax[0].set_xticks(np.arange(0, max(self.DBSCAN_feature_labels)))

        # Plot first variable of original signal
        ax[1].plot(self.df.iloc[:, 0].values, "k", linewidth=0.8)
        ax[1].set_title("Channel 1")
        ax[1].set_ylabel(r"Acceleration [$m/s^2$]")

    def clustering_in_time(self, eps=2, min_samples=1):

        """
        A dataframe containing the information relating to the found impacts. It contains the number of the impact found,
        the time sample index of the start and ending time window the impact occurred in and the approximate time sample
        within this time range that the impact occured.

        Parameters
        ----------
        eps: int
            The radius, size of neighbourhood around each point in the DBSCAN algorithm.

        min_samples:
            The minimum number of samples in a neighbourhood to be considered a 'core' point.

        Returns
        -------
        results_df: pandas.DataFrame
            A dataframe containing the found 'impact_number', its start and end time sample corresponding
            to the time windows the impact occurred in, and the 'approximate_impact_time_sample' that the
            impact occured at.
        """

        DBSCAN_clusters = pd.DataFrame(
            {
                "DBSCAN_labels": self.DBSCAN_feature_labels,
                "window_index": range(len(self.DBSCAN_feature_labels)),
                "window_start": np.zeros(len(self.DBSCAN_feature_labels)),
                "window_end": np.zeros(len(self.DBSCAN_feature_labels)),
                "impact_number": np.zeros(len(self.DBSCAN_feature_labels)),
            }
        )

        # Select only points corresponding to chosen cluster
        clusters_in_time = DBSCAN_clusters[
            DBSCAN_clusters["DBSCAN_labels"] == self.chosen_cluster
        ]

        # Cluster the selected cluster group in time
        DBSCAN_model = DBSCAN(eps=eps, min_samples=1).fit(clusters_in_time.values)
        self.DBSCAN_time_labels = DBSCAN_model.labels_ + 1

        # Find the that an identified cluster in time starts and ends
        clusters_in_time.loc[:, "window_start"] = [
            i[0] for i in self.windowed_index[clusters_in_time["window_index"]]
        ]
        clusters_in_time.loc[:, "window_end"] = [
            i[-1] for i in self.windowed_index[clusters_in_time["window_index"]]
        ]
        clusters_in_time.loc[:, "impact_number"] = self.DBSCAN_time_labels

        # Create dataframe with corresponding impacts
        impulse_list = []

        for impact in clusters_in_time["impact_number"].unique():
            impulse_dict = {
                "impact_number": [],
                "start_time_sample": [],
                "end_time_sample": [],
                "approximate_impact_time_sample": [],
            }
            df_impact_number = clusters_in_time[
                clusters_in_time["impact_number"] == impact
            ]
            impulse_dict["impact_number"] = impact
            impulse_dict["start_time_sample"] = df_impact_number["window_start"].values[
                0
            ]
            impulse_dict["end_time_sample"] = df_impact_number["window_end"].values[-1]
            impulse_list.append(impulse_dict)

        self.results_df = pd.DataFrame(impulse_list)

        for i in range(self.results_df.shape[0]):

            time_window = (
                self.df.iloc[
                    self.results_df.loc[i, "start_time_sample"] : self.results_df.loc[
                        i, "end_time_sample"
                    ],
                    :,
                ]
                .abs()
                .values.T
            )

            approx_time = self.results_df.loc[i, "start_time_sample"]
            +time_window.mean(axis=0).argmax()

            self.results_df.loc[i, "approximate_impact_time_sample"] = approx_time

        return self.results_df


def create_window_indices(array_length: int, window_size: int, stride: int) -> np.array:
    """_summary_

    :param array_length: _description_
    :param window_size: _description_
    :param stride: _description_
    :return: _description_
    """
    windowed_index = utils.windowed_view(
        np.arange(0, array_length).reshape(1, -1), window_size, stride
    )[0]

    return windowed_index


def extract_VGG16_features(
    time_series: np.array,
    window_size: int,
    stride: int,
    image_type: str,
    compression_percentage: float = 1.0,
):
    """
    Splits the given multivariate signal into time windows according to the given window_size and stride parameters.
    The time windows are converted to an image according to the image_type string. The images are compressed and
    passed through the pretrained VGG16 CNN.

    :param :

    """

    # Create array of indices for each time window
    windowed_indices = create_window_indices(time_series.shape[0], window_size, stride)

    # set the input image size for the CNN
    cnn_window_size = np.floor(window_size * compression_percentage)

    # load pretrained CNN model
    model = vgg16.VGG16(
        include_top=False, input_shape=(cnn_window_size, cnn_window_size, 3)
    )
    flat_output = Flatten()(model.layers[-1].output)
    model = Model(inputs=model.inputs, outputs=flat_output)

    VGG16_feature_vectors = []
    for window_i in windowed_indices:

        # Raw data window
        window_data_i = time_series[window_i, :]

        # Apply VGG16 image preprocessing
        input_i = _windowed_VGG16_preprocessing(
            window_data_i, image_type, cnn_window_size
        )

        # extract feature vector
        feature_vector_i = model.predict(input_i)[0]

        VGG16_feature_vectors.append(feature_vector_i)

    return np.array(VGG16_feature_vectors)


def _windowed_VGG16_preprocessing(
    data_window: np.array, image_type: str, window_size: int
) -> np.array:
    """_summary_

    :param data_window: _description_
    :param image_type: _description_
    :param window_size: _description_
    :return: _description_
    """
    # Generate image representing the time window
    if image_type == "none":
        img = np.abs(data_window)
    else:
        img = calculate_recurrence_plot(data_window, image_type)

    # Compress time window
    img = resize(
        img,
        (window_size, window_size),
        anti_aliasing=True,
    )

    # Generate image tensor
    img_3 = np.zeros((img.shape[0], img.shape[1], 3))
    img_3[:, :, 0] = img
    img_3[:, :, 1] = img
    img_3[:, :, 2] = img

    input = np.expand_dims(img_3, axis=0)
    # Apply VGG16 image preprocessing
    input = vgg16.preprocess_input(input)

    return input


def get_PCA_transformed_and_variance(
    feature_vectors: np.array, max_components: int
) -> Tuple[np.array, np.array]:

    """
    Decomposes the feature vectors found through the CNN to max_components number of principle components.

    Parameters
    ----------
    max_components: int
        The number of principle components used in the PCA model.

    """

    pca_model = PCA(n_components=max_components)

    pca_transformed = pca_model.fit_transform(feature_vectors)

    return pca_transformed, pca_model.explained_variance_ratio_


def plot_PCA_explained_variance(explained_variance: np.array) -> plt.fig:
    """_summary_

    :param explained_variance: _description_
    :return: _description_
    """
    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 5))

    plt.plot(np.cumsum(explained_variance), marker=".", c="k")
    plt.xticks(range(explained_variance.shape[0]))
    ax.set_xticklabels(np.arange(explained_variance.shape[0]) + 1)
    plt.title("PCA Components cumulative  variance")
    plt.xlabel("No. of components")
    plt.ylabel("Variance")

    return fig


def get_UMAP_tranformed(
    feature_vectors: np.array,
    n_neighbors: int,
    min_dist: float,
    n_components: int = 2,
    random_state: int = 1,
) -> np.array:
    """_summary_

    :param feature_vectors: _description_
    :param n_neighbors: _description_
    :param min_dist: _description_
    :param n_components: _description_, defaults to 2
    :param random_state: _description_, defaults to 1
    :return: _description_
    """

    umap_model = umap_decomposition.UMAP(
        n_neighbors, min_dist, random_state, n_components
    )

    umap_transformed = umap_model.fit_transform(feature_vectors)

    return umap_transformed


def plot_UMAP_2D(
    UMAP_coords: np.array, kwargs: dict = {"c": "k", "marker": "."}
) -> plt.figure:

    plt.cla()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(UMAP_coords[:, 0], UMAP_coords[:, 1], **kwargs)
    ax.set_title("UMAP decomposition")

    return fig
