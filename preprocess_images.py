import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

from pandarallel import pandarallel

from random import randrange
from matplotlib.image import imread


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
