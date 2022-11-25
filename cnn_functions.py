import numpy as np

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

from tqdm import tqdm_notebook

global seed
seed = 42


def predict_category(img_filename, img_dir, model, dict_categories):
    """
    :param img_filename:
    :param img_dir:
    :param model:
    :param dict_categories:
    :return:
    """
    img = prepare_image_as_input(img_filename, img_dir)

    # predict
    y = model.predict(img)
    # print(y)
    # print("y argmax", y.argmax())
    # get category
    category_nb = y.argmax()
    category = dict_categories[category_nb]
    return category_nb, category


def map_category(y, dict_categories):
    """

    :param y: y is a proba vector of length 7
    :param dict_categories:
    :return:
    """
    return dict_categories[y.argmax()]


def prepare_image_as_input(img_filename, img_dir):
    """
    preprocess the image for VGG
    :param img_filename:
    :param img_dir:
    :return:
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

    :param img_filename:
    :param img_dir:
    :param model:
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
    :param img_path_column_name: (string)
    :param img_dir: (string)
    :param model:
    :return:
    """
    all_features = []
    for index in tqdm_notebook(range(df.shape[0])):
        img_filename = df.at[index, img_path_column_name]
        features = predict_features(img_filename, img_dir, model)
        all_features.append(features)

    features_by_img = np.asarray(all_features, dtype=object)
    features_all = np.concatenate(all_features, axis=0)
    return features_by_img, features_all


def build_vgg_features():
    """

    :return:
    """
    # load model
    vgg = VGG16()
    # remove the output layer
    vgg_features_model = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    return vgg_features_model
