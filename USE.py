import numpy as np
from time import time


def feature_USE_fct(embed, sentences, b_size):
    """

    :param embed:
    :param sentences:
    :param b_size:
    :return:
    """
    batch_size = b_size
    time1 = time()

    for step in range(len(sentences) // batch_size):
        idx = step * batch_size
        feat = embed(sentences[idx:idx + batch_size])

        if step == 0:
            features = feat
        else:
            features = np.concatenate((features, feat))

    time2 = np.round(time() - time1, 0)
    print("temps traitement : ", time2)
    return features
