import src.slam_detector.slam_detector as sut
import numpy as np
import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_swiss_roll


def test_create_window_indices():

    array_length = 10
    window_size = 5
    stride = 5
    window_indices = sut.create_window_indices(array_length, window_size, stride)

    assert_array_equal(window_indices, [np.arange(5), np.arange(5, 10)])


def test_windowed_VGG16_preprocessing():

    data_window = np.random.rand(12, 50)
    compressed_size = 25
    processed_img = sut._windowed_VGG16_preprocessing(
        data_window, "euclidean", compressed_size
    )

    assert processed_img[0].shape[0] == compressed_size
    assert processed_img[0].shape[1] == compressed_size
    assert processed_img[0].shape[2] == 3


def test_get_PCA_transformed_and_variance():
    feature_vectors = np.random.rand(100, 12)
    feature_vectors[:, 10::] = 0
    pca_transformed, explained_variance = sut.get_PCA_transformed_and_variance(
        feature_vectors, 12
    )

    assert all(explained_variance[10::] == 0)


def test_plot_PCA_explained_variance():

    feature_vectors = np.random.rand(100, 12)
    feature_vectors[:, 10::] = 0
    pca_transformed, explained_variance = sut.get_PCA_transformed_and_variance(
        feature_vectors, 12
    )
    fig = sut.plot_PCA_explained_variance(explained_variance)
    file_path = "tests/tests_output/"
    plt.savefig(file_path + "PCA_explained_variance_plot.png")


def test_get_UMAP_tranformed():

    feature_vectors = np.random.rand(100, 12)
    n_components = 3
    umap_transformed = sut.get_UMAP_tranformed(feature_vectors, 5, 0.2, n_components)

    assert umap_transformed.shape[0] == feature_vectors.shape[0]
    assert umap_transformed.shape[1] == n_components


def test_plot_UMAP_2D():

    n_samples = 500
    noise = 0.01
    feature_vectors, y = make_swiss_roll(n_samples, noise=noise, random_state=10)

    n_components = 2
    umap_transformed = sut.get_UMAP_tranformed(feature_vectors, 20, 1, n_components)

    fig = sut.plot_UMAP_2D(umap_transformed, {"c": y})
    file_path = "tests/tests_output/"
    plt.savefig(file_path + "UMAP_plot.png")


def test_plot_compare_clusters_to_timeseries():

    timeseries = np.random.rand(3, 100)
    cluster_labels = np.ones(100)

    fig = sut.plot_compare_clusters_to_timeseries(cluster_labels, timeseries)
    file_path = "tests/tests_output/"
    plt.savefig(file_path + "cluster_evaluation_plot.png")
