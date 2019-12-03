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

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

"""
Goal: pretend like we know something about AI programming....
   
Description: 
    
"""
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set a random seed so runs are repeatable
np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
tf.random.set_seed(98234)  # and tensorflow graph seed

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param y_true:
    :param y_pred:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
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


def run_analysis(predictions, test_labels, out_dir):
    """

    :param predictions:
    :param test_labels:
    :param out_dir:
    :return:
    """
    logging.info("Started analysis")

    # write the confusion_matrix to a file in output
    with open(out_dir + '/confusion.matrix', 'w') as f:
        f.write(np.array2string(confusion_matrix(test_labels, predictions), separator=', '))

    print(accuracy_score(test_labels, predictions))

    # plot the confusion matrix for windows
    if os.name == 'nt':
        class_name = ['0', '1', '2', '3']
        np.set_printoptions(precision=2)
        plot_confusion_matrix(test_labels, predictions, classes=class_name, title='Confusion matrix, without normalization')
        plot_confusion_matrix(test_labels, predictions, classes=class_name, normalize=True, title='Normalized confusion matrix')

        plt.show()

    logging.info("Finished analysis")


def test_model(model, test_data, test_labels):
    """

    :param test_labels:
    :param model:
    :param test_data:
    :return predictions: This should be a numpy array
    """
    logging.info("Started test model")

    loss, acc = model.evaluate(test_data, test_labels, verbose=2)
    logging.info("Testing evaluation of model, accuracy: {:5.2f}%".format(100 * acc))

    """
    predictions = model.perdict(test_data,
                                batch_size=32,
                                verbose=0,
                                steps=None,
                                callbacks=None,
                                max_queue_size=16,
                                workers=8,
                                use_multiprocessing=False)
    """
    predictions = model.predict_classes(test_data, verbose=1)
    logging.info("Finished test model")
    return predictions


def train_model(model, out_model, train_data, train_labels):
    """

    :param use_files:
    :param model:
    :param out_model:
    :param train_data:
    :param train_labels:
    :return:
    """
    logging.info("Started train model")

    loss, acc = model.evaluate(train_data, train_labels, verbose=2)
    logging.info("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    checkpoint_path = out_model + "/checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if os.path.exists(checkpoint_dir):
        # model.load("Model")
        model.load_weights(checkpoint_path)
        logging.info("Model Loaded from files")
    else:
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=1)

        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=.10)
        model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_data=(val_data, val_labels), callbacks=[cp_callback])
        # model.fit(train_data, train_labels, batch_size=32, epochs=10, callbacks=[cp_callback])
        # model.save("Model")
        logging.info("Model saved to files")

    loss, acc = model.evaluate(train_data, train_labels, verbose=2)
    logging.info("Trained model, accuracy: {:5.2f}%".format(100 * acc))

    logging.info("Finished train model")

    return model


def build_model(input_shape):
    """
    What are the weights, optimizer, etc.??

    :param input_shape: Dynamic based on image data
    :return:
    """
    logging.info("Started build model")

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu,
                                  input_shape=(input_shape[1], input_shape[2], input_shape[3])))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))

    # print the output
    # model.summary()

    # are more layers better?

    # Add Dense Layers
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(2048, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    # model.add(keras.layers.Dense(64, activation=tf.nn.softmax)
    # model.add(keras.layers.Dense(32, activation=tf.nn.softmax))
    model.add(keras.layers.Dense(16, activation=tf.nn.softmax))
    model.add(keras.layers.Dense(5, activation=tf.nn.softmax))

    # print the output
    model.summary()

    # adam optimizer
    adam = keras.optimizers.Adam(lr=0.0001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 decay=0.0,
                                 amsgrad=False)

    # Compile
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    logging.info("Finished build model")

    # return model created
    return model


def colorize_mask_(mask, color_map=None):
    """
    Attaches a color palette to a PIL image. So long as the image is saved as a PNG, it will render visibly using the
    provided color map.

    This could be used to identify the different colors of damage within an image.

    :param mask: PIL image whose values are only 0 to 4 inclusive
    :param color_map: np.ndarray or list of 3-tuples with 5 rows
    :return:
    """
    color_map = color_map or np.array([(0, 0, 0),  # 0=background
                                       (255, 0, 0),  # no damage (or just 'building' for localization)
                                       (0, 255, 0),  # minor damage
                                       (0, 0, 255),  # major damage
                                       (128, 128, 0),  # destroyed
                                       ])
    assert color_map.shape == (5, 3)
    mask.putpalette(color_map.astype(np.uint8))
    return None


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

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        logging.info("output directory created %s", out_dir)

    if use_files & os.path.isfile(out_dir + '/images.npy'):
        logging.info("loading np arrays from files: %s.", out_dir + '/images.npy & label.npy')
        images_array = np.load(out_dir + '/images.npy')
        labels_array = np.load(out_dir + '/label.npy')
        logging.info("np arrays loaded from files: %s.", out_dir + '/images.npy & label.npy')
    else:
        # If you wanted to Hard coded Path for training data...
        # data_dir = "/path to data d"

        image_paths = []
        image_paths.extend([(data_dir + "/" + pic) for pic in os.listdir(data_dir)])

        images = []
        labels = []

        for img_path in tqdm(image_paths):

            img_obj = Image.open(img_path)

            # resize the image
            image_shape = (512, 512)
            img_obj = img_obj.resize(image_shape)

            img_array = np.array(img_obj)
            images.append(img_array)

            # Get corresponding label for the current image
            label_path = img_path.replace('png', 'json').replace('images', 'labels')

            # if we don't have label file, continue to the next file.
            if not os.path.exists(label_path):
                continue

            label_file = open(label_path)
            label_data = json.load(label_file)
            damage_encoder = 0

            for feat in label_data['features']['xy']:

                # only images post-disaster will have damage type
                try:
                    damage_type = feat['properties']['subtype']
                    if damage_type != "no-damage":
                        # get the max damage for the image
                        if damage_intensity_encoding[damage_type] > damage_encoder:
                            damage_encoder = damage_intensity_encoding[damage_type]
                        # break

                except:  # pre-disaster damage is default no-damage
                    if damage_intensity_encoding["no-damage"] > damage_encoder:
                        damage_encoder = damage_intensity_encoding["no-damage"]
                    # continue

            labels.append(damage_encoder)

        images_array = np.asarray(images)
        labels_array = np.asarray(labels)
        logging.info("arrays converted to numpy arrays")

        if use_files:
            # Save output file
            np.save(out_dir + '/label.npy', labels_array)
            np.save(out_dir + '/images.npy', images_array)
            logging.info("np arrays saved %s.", out_dir + '/label.npy & /images.npy')

    logging.info("Finished Load JSON and Image into numpy")
    return images_array, labels_array


def main():
    """
    Main Method to parse the inputs, and methods to feed in json data, load images into numpy array, and build/train/test model.
    :return:
    """

    parser = argparse.ArgumentParser(description='CS4793 Training Model')
    parser.add_argument('--data',
                        default='C:/Dev/Workspaces/Python/CS4793/data',
                        # Malay's dir
                        # default='D:/Fall2019/AI/train',
                        metavar="/home/scratch1/cs4793/data",
                        help="Full path to the parent data directory")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.1,
                        metavar='Percentage to use for validation',
                        help="Percentage to use for validation")
    parser.add_argument('--out',
                        default='C:/Dev/Workspaces/Python/CS4793/data/out',
                        # Malay's Dir
                        # default='D:/Fall2019/AI/train/out',
                        metavar='Output directory',
                        help="Output directory")
    parser.add_argument('--use_numpy_files',
                        default=True,
                        # Malay's Dir
                        # default='D:/Fall2019/AI/train/out',
                        metavar='Should the program try to save and load npy files for the image and label np arrays.',
                        help="True or False: Designate if the program should try to save and load npy files")
    args = parser.parse_args()

    out_training = args.out + "/train"
    out_testing = args.out + "/test"
    out_model = args.out + "/model"

    # load the Training images and labels
    image_array, label_array = load_json_and_img(args.data + "/train/images", out_training, args.use_numpy_files)

    # build the model
    model = build_model(image_array.shape)

    # train the model
    model = train_model(model, out_model, image_array, label_array)

    # load the Testing images and labels (there will be no labels)
    test_data, test_labels = load_json_and_img(args.data + "/test/images", out_testing, args.use_numpy_files)

    test_data = tf.dtypes.cast(test_data, tf.float32)

    # test the model
    predictions = test_model(model, test_data, test_labels)

    # do some analysis
    run_analysis(predictions, test_labels, args.out)


if __name__ == "__main__":
    main()
