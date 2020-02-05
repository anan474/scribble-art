import configparser
import argparse
import subprocess
import os
import cv2
import numpy as np
import random
import math


def get_prepared_image(source_image, scale_factor):
    """
    The original image must be resized and turned into a
    grayscale image.
    """
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, dsize=(0,0), fx=scale_factor, fy=scale_factor)
    return resized_image



def create_scribble_art(config):
    source_image = cv2.imread(config["INPUT_OUTPUT"]["input_image"])
    scale_factor = float(config["DRAWING"]["image_scale_factor"])
    prepared_image = get_prepared_image(source_image, scale_factor)
    cv2.imshow("test", prepared_image)
    cv2.waitKey(0)

def delete_and_create_output_folder():
    """
    Each time the program is run, the previous output
    folder shall be deleted.
    """
    with open(os.devnull, 'wb') as quiet_output:
        subprocess.call(["rm", "-r", "output"])
        subprocess.call(["mkdir", "-p", "output"])


def get_config(filename):
    """
    All settings are stored in an external text file.
    """
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def main():
    """
    The program start and ends here.
    """
    delete_and_create_output_folder()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        default="options.cfg",
        help="path to program options file")
    arguments = vars(parser.parse_args())
    filename = arguments["config"]
    config = get_config(filename)
    create_scribble_art(config)


if __name__ == '__main__':
    main()
