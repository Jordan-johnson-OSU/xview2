"""
CS4793 - Working on xview2 data challenge.

https://xview2.org/

assessing building damage after a natural disaster

@authors: Melanie Bischoff, Malay Bhakta, Jordan Johnson
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import sys
import itertools

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set a random seed so runs are repeatable
np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
# tf.random.set_seed(98234)  # and tensorflow graph seed

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0

"""
Goal: pretend like we know something about AI programming....

Description: 
after 

"""


def run_analysis():
    """
    Classification ranking?  Print out stuff?
    :return:
    """
    logging.info("Started analysis")
    logging.info("Finished analysis")


def test_model(model, test_data, test_labels):
    """

    :param model:
    :param test_data:
    :param test_labels:
    :return:
    """
    logging.info("Started test model")
    model.evaluate(test_data, test_labels, batch_size=32)
    logging.info("Finished test model")
    return model


def load_json_and_img(data_dir, out_dir, use_files):
    """
    Assumptions about the data are that it looks like ../data/train/{damage}/images and ../data/train/{damage}/labels

    :param use_files:  Boolean True or False to try to save and load npy files.  Do not use on server.
    :param out_dir:
    :param data_dir:
    :return: images, labels
    """
    logging.info("Started Load JSON and Image into numpy")
    images_array = np.array([])
    labels_array = np.array([])

    if use_files & os.path.isfile(out_dir + '/images.npy'):
        logging.info("loading np arrays from files.")
        images_array = np.load(out_dir + '/images.npy')
        labels_array = np.load(out_dir + '/label.npy')
        logging.info("np arrays loaded from files.")
    else:
        # data_dir = data_dir + "/train/images"

        # If you wanted to Hard coded Path for training data...
        # data_dir = "C:/Dev/Workspaces/Python/AI Learning/program/data/train/images"

        image_paths = []
        image_paths.extend([(data_dir + "/" + pic) for pic in os.listdir(data_dir)])

        images = []
        labels = []

        for img_path in tqdm(image_paths):

            img_obj = Image.open(img_path)
            # resize the image
            IMAGE_SHAPE = (512, 512)
            img_obj = img_obj.resize(IMAGE_SHAPE)

            img_array = np.array(img_obj)
            images.append(img_array)

            # Get corresponding label for the current image
            label_path = img_path.replace('png', 'json').replace('images', 'labels')
            label_file = open(label_path)
            label_data = json.load(label_file)
            damage_encoder = 0

            for feat in label_data['features']['xy']:

                # only images post-disaster will have damage type
                try:
                    damage_type = feat['properties']['subtype']
                    if damage_type != "no-damage":
                        damage_encoder = damage_intensity_encoding[damage_type]
                        break

                except:  # pre-disaster damage is default no-damage
                    damage_encoder = damage_intensity_encoding["no-damage"]
                    continue

            labels.append(damage_encoder)

        images_array = np.asarray(images)
        labels_array = np.asarray(labels)
        logging.info("arrays converted to numpy arrays")

        if use_files:
            # Save output file
            np.save(out_dir + '/label.npy', labels_array)
            np.save(out_dir + '/images.npy', images_array)
            logging.info("np arrays saved.")

    logging.info("Finished Load JSON and Image into numpy")
    return images_array, labels_array


def load_test_img(data_dir, out_dir, use_files):
    logging.info("Started Load JSON and Image into numpy")
    images_array = np.array([])
    labels_array = np.array([])

    if use_files & os.path.isfile(out_dir + '/images.npy'):
        logging.info("loading np arrays from files.")
        images_array = np.load(out_dir + '/images.npy')
        labels_array = np.load(out_dir + '/label.npy')
        logging.info("np arrays loaded from files.")
    else:
        # data_dir = data_dir + "/train/images"

        # If you wanted to Hard coded Path for training data...
        # data_dir = "C:/Dev/Workspaces/Python/AI Learning/program/data/train/images"

        image_paths = []
        image_paths.extend([(data_dir + "/" + pic) for pic in os.listdir(data_dir)])

        images = []
        labels = []

        for img_path in tqdm(image_paths):

            img_obj = Image.open(img_path)
            # resize the image
            IMAGE_SHAPE = (512, 512)
            img_obj = img_obj.resize(IMAGE_SHAPE)

            img_array = np.array(img_obj)
            images.append(img_array)

            # Get corresponding label for the current image
            # Get corresponding label for the current image
            label_path = img_path.replace('png', 'json').replace('images', 'labels')
            label_file = open(label_path)
            label_data = json.load(label_file)
            damage_encoder = 0

            for feat in label_data['features']['xy']:

                # only images post-disaster will have damage type
                try:
                    damage_type = feat['properties']['subtype']
                    if damage_type != "no-damage":
                        damage_encoder = damage_intensity_encoding[damage_type]
                        break

                except:  # pre-disaster damage is default no-damage
                    damage_encoder = damage_intensity_encoding["no-damage"]
                    continue

            labels.append(damage_encoder)

        images_array = np.asarray(images)
        labels_array = np.asarray(labels)

        logging.info("arrays converted to numpy arrays")

        if use_files:
            # Save output file
            np.save(out_dir + '/label.npy', labels_array)
            np.save(out_dir + '/images.npy', images_array)
            logging.info("np arrays saved.")

    logging.info("Finished test Images into numpy")
    return images_array, labels_array


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():
    """
    Main Method to parse the inputs, and methods to feed in json data, load images into numpy array, and build/train/test model.
    :return:
    """

    parser = argparse.ArgumentParser(description='CS4793 Training Model')
    parser.add_argument('--data',
                        # default='C:/Dev/Workspaces/Python/CS4793/xview2/data',
                        # Malay's dir
                        default='D:/Fall2019/AI/train',
                        metavar="/home/scratch1/cs4793/data",
                        help="Full path to the parent data directory")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.1,
                        metavar='Percentage to use for validation',
                        help="Percentage to use for validation")
    parser.add_argument('--out',
                        # default='C:/Dev/Workspaces/Python/CS4793/xview2/out',
                        # Malay's Dir
                        default='D:/Fall2019/AI/train/out',
                        metavar='Output directory',
                        help="Output directory")
    parser.add_argument('--use_numpy_files',
                        default=True,
                        # Malay's Dir
                        # default='D:/Fall2019/AI/train/out',
                        metavar='Should the program try to save and load npy files for the image and label np arrays.',
                        help="True or False: Designate if the program should try to save and load npy files")
    args = parser.parse_args()

    """
    1. Let's just start with seeing if we can train on a model to identify if anything is damaged or not
    2. Identify buildings and what not.
    3. Cake!
    """
    # load the Training images and labels, we only need lables
    # image_array, label_array = load_json_and_img(args.data + "/train/images", args.out + "/train", args.use_numpy_files)
    test_data, test_label = load_test_img(args.data + "/test/Newfolder/images", args.out + "/test",
                                          args.use_numpy_files)

    # build the model
    # Model = name of the model
    model = tf.keras.models.load_model("Model")
    print(test_data.shape, test_label.shape)
    np.set_printoptions(threshold=sys.maxsize)
    print(test_label)
    label1 = LabelEncoder()
    print("=======================")
    test_label = label1.fit_transform(test_label)
    np.set_printoptions(threshold=sys.maxsize)
    print(test_label)

    # load the Testing images and labels
    # TODO: what are the test_labels?
    # test_data, test_labels = load_json_and_img(args.data + "/test/images", args.out + "/test", args.use_numpy_files)
    """
    predict = model.predict(test_data,
                            batch_size=None,
                            verbose=0,
                            steps=None,
                            callbacks=None,
                            max_queue_size=10,
                            workers=1,
                            use_multiprocessing=True)

    print(predict)
    """
    predict = model.predict_classes(test_data, batch_size = 10, verbose = 1)
    for i in predict:
        print(i)
    # test the model
    # model = test_model(model, test_data, test_labels)
    # model.evaluate(test_data, test_label)
    # do some analysis
    # run_analysis(model)
    mat = confusion_matrix(test_label, predict)

    print(confusion_matrix(test_label, predict))
    print(accuracy_score(test_label, predict))

    class_name = ['0', '1', '2', '3']

    print(mat)



    np.set_printoptions(precision=2)
    plot_confusion_matrix(test_label, predict, classes=class_name, title='Confusion matrix, without normalization')
    plot_confusion_matrix(test_label, predict, classes=class_name, normalize=True, title='Normalized confusion matrix')

    plt.show()

    # plot_confusion_matrix(conf_mat=mat, figsize=(8,8), class_names= class_names, show_normed=True)
    # plt.tight_layout()


if __name__ == "__main__":
    main()
"""
test_images = load_test_images('D:/Fall2019/AI/train/test')
test_resize_images = tf.compat.v2.image.resize_with_pad(
    test_images,
    512,
    512,
    # method=ResizeMethod.BILINEAR
)

train_images, train_labels = load_json_and_img('D:/Fall2019/AI/train')

#loading the model with name in " " and in same directory.
model.summary()

print(train_labels)
label1 = LabelEncoder()
train_labels = label1.fit_transform(train_labels)
print(train_labels)

predict = model.predict(train_labels)
print(predict)
"""
