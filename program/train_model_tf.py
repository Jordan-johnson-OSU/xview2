"""
CS4793

@authors: Melanie Bischoff, Malay Bhakta, Jordan Johnson
"""

import sys
import json
import os
import math
import random
import argparse
import logging
import platform
import cv2
import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

logging.basicConfig(level=logging.INFO)

# Set a random seed so runs are repeatable
np.random.seed(98234)  # need to set numpy seed since np.random.shuffle is used
tf.random.set_seed(98234)  # and tensorflow graph seed

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

def test_model(model):
    """

    :param model:
    :return:
    """
    return model

def train_model(model, train_data, train_labels, val_data, val_labels):
    """
    look at the the xview2/baseline/model/damage_classification.py and xview2/baseline/model/model.py

    :param model:
    :return: model
    """

    model.fit(train_data, train_labels, epochs=10,
              validation_data=(val_data, val_labels))

    return model

def build_model(data_dir):
    """
    look at the the xview2/baseline/model/damage_classification.py and xview2/baseline/model/model.py
    CNN?
    How many layers?  What weights?  What method? Optimizer, etc.

    :param data_dir:
    :return: model
    """

    #TODO: is this the right thing to start on?
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #is this the right shape?
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    #print the output
    model.summary()

    #are more layers better?

    #Add Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))

    #print the output
    model.summary()

    #TODO: add weights and generators and optimizers

    # adam optimizer
    adam = keras.optimizers.Adam(lr=0.0001,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 decay=0.0,
                                 amsgrad=False)

    #Compile
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #TODO: return model created;
    return model

def load_json_and_img(data_dir):
    """
    Assumptions about the data are that it looks like ../data/train/{damage}/images and ../data/train/{damage}/labels

    :param data_dir:
    :return: something
    """
    data = [folder for folder in os.listdir(data_dir) if
                 not folder.startswith('.') and ('midwest') not in folder]
    paths = ([data_dir + "/train/" + d + "/images" for d in data])
    image_paths = []
    image_paths.extend(
        [(disaster_path + "/" + pic) for pic in os.listdir(disaster_path)] for disaster_path in paths)
    img_paths = np.concatenate(image_paths)

    for img_path in tqdm(img_paths):

        img_obj = Image.open(img_path)
        img_array = np.array(img_obj)

        #Get corresponding label for the current image
        label_path = img_path.replace('png', 'json').replace('images', 'labels')
        label_file = open(label_path)
        label_data = json.load(label_file)

    #TODO: file or struct how to pass the data around?
    return something

def main():
    """
    Main Method to parse the inputs, and methods to feed in json data, load images into numpy array, and build/train/test model.
    :return:
    """

    parser = argparse.ArgumentParser(description='Run XView2 Challenge Disaster Damage Classification Training & Evaluation')
    parser.add_argument('--data',
                        default='../../data',
                        metavar="/home/scratch1/cs4793/data",
                        help="Full path to the parent data directory")
    parser.add_argument('--val_split_pct',
                        required=False,
                        default=0.0,
                        metavar='Percentage to use for validation',
                        help="Percentage to use for validation")
    args = parser.parse_args()

    logging.info("Started Load JSON and Image into numpy")
    something = load_json_and_img(args.data, float(args.val_split_pct))
    logging.info("Finished Load JSON and Image into numpy")

    logging.info("Started Load JSON")
    model = build_model(something)
    logging.info("Finished Load JSON")

    logging.info("Started train model")
    model = train_model(model)
    logging.info("Finished train model")

    logging.info("Started test model")
    model = test_model(model)
    logging.info("Finished test model")

    logging.info("Started analysis")
    run_analysis(model)
    logging.info("Finished analysis")


if __name__ == "__main__":
    main()
