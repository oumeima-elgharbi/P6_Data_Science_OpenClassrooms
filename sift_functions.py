import numpy as np

import cv2  # SIFT

from sklearn.cluster import MiniBatchKMeans

from time import time
from tqdm import tqdm

global seed
seed = 42


def generate_keypoints(df, img_path_column_name, img_dir):
    """

    :param df:
    :param img_path_column_name:
    :param img_dir:
    :return:
    """
    # identification of key points and associated descriptors

    sift_keypoints = []
    temps1 = time()
    sift = cv2.xfeatures2d.SIFT_create(500)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # for image_num in range(len(list_photos)) :
        # if index % 100 == 0 : print(image_num)
        img_path = img_dir + row[img_path_column_name]
        # image = cv2.imread(path_images + list_photos[image_num],0) # convert in gray
        image = cv2.imread(img_path, 0)  # convert in gray
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        res = cv2.equalizeHist(image)  # equalize image histogram
        kp, des = sift.detectAndCompute(res, None)
        sift_keypoints.append(des)

    sift_keypoints_by_img = np.asarray(sift_keypoints)
    sift_keypoints_all = np.concatenate(sift_keypoints_by_img, axis=0)

    print()
    print("Nombre de descripteurs : ", sift_keypoints_all.shape)

    duration1 = time() - temps1
    print("temps de traitement SIFT descriptor : ", "%15.2f" % duration1, "secondes")

    return sift_keypoints_by_img, sift_keypoints_all


def get_bag_of_images(sift_keypoints_all):
    """

    :param sift_keypoints_all:
    :return:
    """

    # Determination number of clusters
    temps1 = time()

    k = int(round(np.sqrt(len(sift_keypoints_all)), 0))
    print("Nombre de clusters estimés : ", k)
    print("Création de", k, "clusters de descripteurs ...")

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=k, init_size=3 * k, random_state=0)
    kmeans.fit(sift_keypoints_all)

    duration1 = time() - temps1
    print("temps de traitement kmeans : ", "%15.2f" % duration1, "secondes")

    return kmeans


def build_histogram(kmeans, des, image_num):
    """
    # Creation of histograms (features)
    :param kmeans:
    :param des:
    :param image_num:
    :return:
    """
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des = len(des)
    if nb_des == 0: print("problème histogramme image  : ", image_num)
    for i in res:
        hist[i] += 1.0 / nb_des
    return hist


def get_images_features(sift_keypoints_by_img, kmeans):
    """

    :param sift_keypoints_by_img:
    :param kmeans:
    :return:
    """
    temps1 = time()

    # Creation of a matrix of histograms
    hist_vectors = []

    for i, image_desc in enumerate(tqdm(sift_keypoints_by_img)):
        # if i % 100 == 0: print(i)
        hist = build_histogram(kmeans, image_desc, i)  # calculates the histogram
        hist_vectors.append(hist)  # histogram is the feature vector

    im_features = np.asarray(hist_vectors)

    duration1 = time() - temps1
    print("temps de création histogrammes : ", "%15.2f" % duration1, "secondes")
    return im_features
