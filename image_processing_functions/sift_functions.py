import warnings
warnings.filterwarnings(action="ignore")

import numpy as np

import cv2  # SIFT

from sklearn.cluster import MiniBatchKMeans

from time import time
from tqdm import tqdm

global seed
seed = 42


def generate_keypoints(df, img_path_column_name, img_dir):
    """

    :param df: (DataFrame)
    :param img_path_column_name: (string)
    :param img_dir: (string)
    :return:
    """
    # identification of key points and associated descriptors

    sift_keypoints = []
    time1 = time()
    sift = cv2.xfeatures2d.SIFT_create(500)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_path = img_dir + row[img_path_column_name]
        # convert in gray
        image = cv2.imread(img_path, 0)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # equalize image histogram
        res = cv2.equalizeHist(image)
        kp, des = sift.detectAndCompute(res, None)
        sift_keypoints.append(des)

    sift_keypoints_by_img = np.asarray(sift_keypoints, dtype=object)
    sift_keypoints_all = np.concatenate(sift_keypoints_by_img, axis=0)

    print()
    print("Number of keypoints : ", sift_keypoints_all.shape)

    duration = time() - time1
    print("SIFT descriptor processing time : ", "%15.2f" % duration, "seconds")

    return sift_keypoints_by_img, sift_keypoints_all


def get_bag_of_images(sift_keypoints_all):
    """

    :param sift_keypoints_all:
    :return:
    """
    time1 = time()

    # Determination number of clusters
    k = int(round(np.sqrt(len(sift_keypoints_all)), 0))
    print("Number of clusters estimated : ", k)
    print("Creating of", k, "clusters of keypoints ...")

    # Clustering
    kmeans = MiniBatchKMeans(n_clusters=k, init_size=3 * k, random_state=0)
    kmeans.fit(sift_keypoints_all)

    duration = time() - time1
    print("KMeans processing time : ", "%15.2f" % duration, "seconds")

    return kmeans


def build_histogram(kmeans, des, image_num):
    """
    Creation of histograms (features)
    :param kmeans:
    :param des:
    :param image_num:
    :return:
    """
    res = kmeans.predict(des)
    hist = np.zeros(len(kmeans.cluster_centers_))
    nb_des = len(des)
    if nb_des == 0:
        print("Problem of image histogram for : ", image_num)
    for i in res:
        hist[i] += 1.0 / nb_des
    return hist


def get_images_features(sift_keypoints_by_img, kmeans):
    """

    :param sift_keypoints_by_img:
    :param kmeans:
    :return:
    """
    time1 = time()

    # Creation of a matrix of histograms
    hist_vectors = []

    for i, image_desc in enumerate(tqdm(sift_keypoints_by_img)):
        # calculates the histogram
        hist = build_histogram(kmeans, image_desc, i)
        # histogram is the feature vector
        hist_vectors.append(hist)

    im_features = np.asarray(hist_vectors)

    duration = time() - time1
    print("Creation of histograms processing time : ", "%15.2f" % duration, "seconds")
    return im_features
