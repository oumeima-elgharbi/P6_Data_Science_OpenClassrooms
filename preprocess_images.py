import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from pandarallel import pandarallel

from random import randrange
from matplotlib.image import imread

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, MiniBatchKMeans

from tqdm import tqdm

from sklearn import manifold, decomposition


global seed
seed = 42

def save_features(path, filename, features):
    """

    :param path:
    :param filename:
    :param features:
    :return:
    """
    np.savez_compressed(path + filename, features)


def load_features(path, filename):
    """

    :param path:
    :param filename:
    :return:
    """
    dict_data = np.load(path + filename, allow_pickle=True)
    features = dict_data['arr_0']
    return features


def list_fct(category, list_img_in_dir, df, img_column_name, label_column_name):
    """

    :param category:
    :param list_img_in_dir:
    :param df:
    :param img_column_name:
    :param label_column_name:
    :return:

    :UC: photos in the local directory must match the photos in the dataset df
    """
    list_images_for_a_category = []
    for photo in list_img_in_dir:  # 1050 photos in our local directory
        # we get the row in our dataset that matches the photo in the local directory
        img_row = df[df[img_column_name] == photo]
        if (img_row[label_column_name] == category).values[0]:  # to np array and get the first element
            list_images_for_a_category.append(photo)
    return list_images_for_a_category


def display_images_per_category(categories, df, path, img_column_name, label_column_name, list_img_in_dir,
                                n_photos_to_display=3):
    """

    :param categories:
    :param df:
    :param path:
    :param img_column_name:
    :param label_column_name:
    :param list_img_in_dir:
    :param n_photos_to_display: (from 2 to 9)
    :return:
    """
    for category in categories:
        print(category)
        print("-------")
        # we get the list of photos for one category
        list_photos_cat = list_fct(category, list_img_in_dir, df, img_column_name, label_column_name)
        n = len(list_photos_cat)
        print("For this category, we have", n, "images.")

        for i in range(n_photos_to_display):
            plt.subplot(int('1{}0'.format(n_photos_to_display)) + 1 + i)

            j = randrange(0, n)  # from 0 to n-1
            filename = path + list_photos_cat[j]  # we choose randomly a photo
            img = imread(filename)
            plt.imshow(img)

        plt.show()


def add_cluster_tsne(df_tsne, cls):
    """

    :param df_tsne:
    :param cls:
    :return:
    """
    df_tsne_cls = df_tsne.copy()

    df_tsne_cls["cluster"] = cls.labels_
    print(df_tsne_cls.shape)
    return df_tsne_cls


def clustering_tsne(X_tsne):
    """

    :param X_tsne:
    :return:
    """
    cls = KMeans(n_clusters=7, random_state=6)
    cls.fit(X_tsne)
    return cls


def display_tsne(df_tsne, column_name):
    """

    :param df_tsne:
    :param column_name:
    :return:
    """
    plt.figure(figsize=(8, 5))  # (10, 6)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue=column_name, data=df_tsne,
        legend="brief", s=50, alpha=0.6)  # palette=sns.color_palette('tab10', n_colors=4),

    plt.title('TSNE selon {}'.format(column_name), fontsize=30, pad=35, fontweight='bold')
    plt.xlabel('tsne1', fontsize=26, fontweight='bold')
    plt.ylabel('tsne2', fontsize=26, fontweight='bold')
    plt.legend(prop={'size': 14})

    plt.show()


def print_ari_score(labels, cls):
    """

    :param labels:
    :param cls:
    :return:
    """
    print("ARI : ", adjusted_rand_score(labels, cls.labels_))


def get_pca_for_features(features):
    """


    :param features:
    :return:
    """
    print("Dimensions dataset avant réduction PCA : ", features.shape)
    pca = decomposition.PCA(n_components=0.99)
    feat_pca = pca.fit_transform(features)
    print("Dimensions dataset après réduction PCA avec 99% variance expliquée: ", feat_pca.shape)
    return feat_pca


def get_tsne(features_pca, df, category_column_name):
    """

    :param features_pca:
    :param df:
    :param category_column_name:
    :return:
    """
    tsne = manifold.TSNE(n_components=2, perplexity=30,
                         n_iter=2000, init='random', random_state=seed)
    X_tsne = tsne.fit_transform(features_pca)

    df_tsne = pd.DataFrame(X_tsne[:, 0:2], columns=['tsne1', 'tsne2'])
    df_tsne["class"] = df[category_column_name]
    print(df_tsne.shape)
    return X_tsne, df_tsne