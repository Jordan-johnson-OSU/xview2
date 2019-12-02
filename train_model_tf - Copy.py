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
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tqdm import tqdm
import h5py


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Set a random seed so runs are repeatable
np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
#tf.random.set_seed(98234)  # and tensorflow graph seed

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


def train_model(model, train_data, train_labels):
    """

    :param model:
    :param train_data:
    :param train_labels:
    :return:
    """
    logging.info("Started train model")

    label1 = LabelEncoder()
    train_labels = label1.fit_transform(train_labels)
    print(train_labels)
    print(train_data.shape, train_labels.shape)
    history = model.fit(train_data, train_labels, batch_size=64, epochs=1)
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    """
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(train_data[0])
    plt.show(block=True)
    """
    #history = model.fit(train_data, train_labels, batch_size=64, validation_split=0.2, epochs=10)
    print(history.history)
    """
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=.15)
    
    # TODO: save the split data here? and pre load next time?
    
    model.fit(train_data, train_labels, epochs=10,
         validation_data=(val_data, val_labels))
    """
    model.save('Model')
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
    """
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add( keras.layers.Dense(5, activation=tf.nn.softmax))
    """

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                  input_shape=(input_shape[1], input_shape[2], input_shape[3])))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    # print the output
    # model.summary()

    # are more layers better?

    # Add Dense Layers
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(2048, activation='relu'))
    #model.add(keras.layers.Dense(1024, activation='relu'))
    #model.add(keras.layers.Dense(512, activation='relu'))
    #model.add(keras.layers.Dense(128, activation='relu'))
    #model.add(keras.layers.Dense(64, activation='softmax'))
    #model.add(keras.layers.Dense(32, activation='softmax'))
    model.add(keras.layers.Dense(16, activation='softmax'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    # print the output
    model.summary()

    # TODO: add weights and generators and optimizers

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
    i = 1
    if use_files & os.path.isfile(out_dir + '/train/images.npy'):
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
            if i > 60:
                break;
            #i += 1
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


def main():
    """
    Main Method to parse the inputs, and methods to feed in json data, load images into numpy array, and build/train/test model.
    :return:
    """

    parser = argparse.ArgumentParser(description='CS4793 Training Model')
    parser.add_argument('--data',
                        #default='C:/Dev/Workspaces/Python/CS4793/xview2/data',
                        # Malay's dir
                        default='C:/Users/melan/OneDrive/Documents/Artificial Intelligence',
                        metavar="/home/scratch1/cs4793/data",
                        help="Full path to the parent data directory")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.1,
                        metavar='Percentage to use for validation',
                        help="Percentage to use for validation")
    parser.add_argument('--out',
                        #default='C:/Dev/Workspaces/Python/CS4793/xview2/out',
                        # Malay's Dir
                        default='C:/Users/melan/OneDrive/Documents/Artificial Intelligence/out',
                        metavar='Output directory',
                        help="Output directory")
    parser.add_argument('--use_numpy_files',
                        default=True,
                        # Malay's Dir
                        #default='D:/Fall2019/AI/train/out',
                        metavar='Should the program try to save and load npy files for the image and label np arrays.',
                        help="True or False: Designate if the program should try to save and load npy files")
    args = parser.parse_args()

    """
    1. Let's just start with seeing if we can train on a model to identify if anything is damaged or not
    2. Identify buildings and what not.
    3. Cake!
    """

    # load the Training images and labels
    image_array, label_array = load_json_and_img(args.data + "/train/images", args.out + "/train", args.use_numpy_files)

    # build the model
    model = build_model(image_array.shape)

    # train the model
    model = train_model(model, image_array, label_array)

    # load the Testing images and labels
    # TODO: what are the test_labels?
    # test_data, test_labels = load_json_and_img(args.data + "/test/images", args.out + "/test", args.use_numpy_files)

    # test the model / predict
    # model = test_model(model, test_data, test_labels)

    # do some analysis
    # run_analysis(model)


if __name__ == "__main__":
    main()
