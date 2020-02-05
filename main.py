import configparser
import argparse
import subprocess
import os
import cv2
import numpy as np
import random
import math



def create_scribble_art(config):
    source_image = cv2.imread(config["INPUT_OUTPUT"]["input_image"])


def delete_and_create_output_folder():
    with open(os.devnull, 'wb') as quiet_output:
        subprocess.call(["rm", "-r", "output"])
        subprocess.call(["mkdir", "-p", "output"])


def get_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config



def main():
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
