# to compute time of pipeline
from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")

global seed
seed = 42


def ARI_fct(features, l_cat, y_cat_num):
    """
    Computes t-SNE, creates clusters and computes ARI score between the real labels / categories and the clusters.

    :param features:
    :param l_cat:
    :param y_cat_num:
    :return:
    """
    time1 = time()
    num_labels = len(l_cat)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000,
                init='random', learning_rate=200, random_state=seed)
    X_tsne = tsne.fit_transform(features)

    # Détermination des clusters à partir des données après Tsne
    cls = KMeans(n_clusters=num_labels, n_init=100, random_state=seed)
    cls.fit(X_tsne)
    ARI = np.round(adjusted_rand_score(y_cat_num, cls.labels_), 4)
    time2 = np.round(time() - time1, 0)
    print("ARI : {}, time : {} seconds.".format(ARI, time2))

    return ARI, X_tsne, cls.labels_


def TSNE_visu_fct(X_tsne, y_cat_num, l_cat, labels, ARI):
    """
    Visualization of t-SNE according to the real categories / clusters.

    :param X_tsne:
    :param y_cat_num: (Pandas Series) the number of the category for each row of X_tsne
    :param l_cat: (list) a list of unique categories
    :param labels:
    :param ARI:
    :return:
    """
    fig = plt.figure(figsize=(15, 6))

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Categorie")
    plt.title('Representation of the real categories.')

    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Representation of the clusters.')

    plt.show()
    print("ARI : ", ARI)
