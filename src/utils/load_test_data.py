from scipy.io import loadmat
import numpy as np
import pandas as pd


def load_test_data(region: int, cap: int) -> pd.DataFrame:

    """
    Loads test data from the associated file it is stored in.

    param region: The chosen region according to the experimental test.
    param cap: The chosen cap number according to the experimental test.
    return df: A dataframe containing the experimental data with the columns corresponding to the measured channels.
    """

    file_path = f"Experiment_Data/Region_{region}_Cap_{cap}_9600Hz.MAT"
    mat_dict = loadmat(file_path)
    channels = ["Channel_%d_Data" % i for i in range(2, 12)]
    channel_data_list = []
    for i in channels:
        channel_data_list.append(mat_dict[i])

    if (
        file_path == "Experiment_Data/Region_2_Cap_2_9600Hz.MAT"
    ):  # There is an extra impulse in this test
        dataset = np.array(channel_data_list)[:, 0:265000, 0]
    else:
        dataset = np.array(channel_data_list)[:, :, 0]

    columns = ["Channel_%d" % i for i in range(1, 11)]
    df = pd.DataFrame(dataset.T, columns=columns)

    return df
