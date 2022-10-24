import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to compute time of pipeline
from time import time, strftime, gmtime

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")


def load_data(path, filename):
    """
    Step 0)
    :param path:
    :param filename: (string)
    :return:
    """
    print("___Loading raw dataset___")

    # Load raw data
    dataset_file = "{}{}".format(path, filename)
    dataset = pd.read_csv(dataset_file)

    print("Initial shape :", dataset.shape)
    return dataset
