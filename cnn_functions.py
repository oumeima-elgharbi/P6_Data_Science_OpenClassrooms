import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from pandarallel import pandarallel

from random import randrange
from matplotlib.image import imread

from tqdm import tqdm

from keras_preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model

from keras.optimizers import SGD

from tqdm import tqdm


global seed
seed = 42

def predict_category(img_filename, img_dir, model, dict_categories):
    """
    y is a proba vector of length 7
    :param y:
    :return:
    """
    img = prepare_image_as_input(img_filename, img_dir)

    # predict
    y = model.predict(img)
    #print(y)
    #print("y argmax", y.argmax())
    # get category
    category_nb = y.argmax()
    category = dict_categories[category_nb]
    return category_nb, category

def map_category(y, dict_categories):
    """
    y is a proba vector of length 7
    :param y:
    :return:
    """
    return dict_categories[y.argmax()]


def prepare_image_as_input(img_filename, img_dir):
    """"
    preprocess the image for VGG
    """
    # get the image path
    img_path = img_dir + img_filename

    # load an image from file
    img = load_img(img_path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    img = img_to_array(img)
    # reshape data for the model # Créer la collection d'images (un seul échantillon)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # prepare the image for the VGG model # Prétraiter l'image comme le veut VGG-16
    img = preprocess_input(img)
    return img


def predict_features(img_filename, img_dir, model):
    """
    y is a proba vector of length 7
    :param y:
    :return:
    """
    # get the image path
    img = prepare_image_as_input(img_filename, img_dir)
    # predict
    features = model.predict(img)
    return features


def get_features(df, img_path_column_name, img_dir, model):
    """

    :param df:
    :param img_path_column_name:
    :param model:
    :return:
    """
    all_features = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_filename = df.at[index, img_path_column_name]
        features = predict_features(img_filename, img_dir, model)
        all_features.append(features)

    features_by_img = np.asarray(all_features)
    features_all = np.concatenate(all_features, axis=0)
    return features_by_img, features_all

