import configparser
import argparse
import subprocess
import os
import cv2
import numpy as np
import random
import math
import connections

def get_empty_white_canvas(size_x=1920, size_y=1080):
    img = np.array([255], dtype=np.uint8) * np.ones((size_y,size_x,1), dtype=np.uint8)
    return img


def get_prepared_image(source_image, scale_factor):
    """
    The original image must be resized and turned into a
    grayscale image.
    """
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, dsize=(0,0), fx=scale_factor, fy=scale_factor)
    return resized_image


def get_point_thresholds(no_of_layers, exponent, prefactor):
    return [math.pow(x, exponent) * prefactor for x in range(no_of_layers)]


def get_layer_points(current_max, point_threshold, prepared_image):
    """
    This function returns the points we will
    later connect with line segments.
    Random points may be put anywhere where the lightness
    of the image is below current_max. For each
    pixel that may potentially represent a point, we draw
    a random number. A point is only created if the random number
    if below point_threshold.
    """
    white_value = 255
    layer = prepared_image.copy()
    # find all positions where the image is darker than current_max
    layer[layer <= current_max] = 0
    layer[layer != 0] = white_value

    random_matrix = np.random.rand(*prepared_image.shape)
    layer[random_matrix > point_threshold] = white_value
    points = np.argwhere(layer == 0)
    points_tuples = [(p[1],p[0]) for p in points]
    return points_tuples

def create_scribble_art(config):
    source_image = cv2.imread(config["INPUT_OUTPUT"]["input_image"])
    scale_factor = float(config["DRAWING"]["image_scale_factor"])
    prepared_image = get_prepared_image(source_image, scale_factor)

    no_of_layers = int(config["DRAWING"]["no_of_layers"])
    exponent = float(config["DRAWING"]["point_thresholds_exponent"])
    prefactor = float(config["DRAWING"]["point_thresholds_prefactor"])
    point_thresholds = get_point_thresholds(no_of_layers, exponent, prefactor)
    gray_value_step = 255.0 / len(point_thresholds)

    canvas = get_empty_white_canvas(prepared_image.shape[1], prepared_image.shape[0])
    max_distance = float(config["DRAWING"]["max_line_length_factor"]) * min(canvas.shape[1], canvas.shape[0])
    for layer_index in range(no_of_layers):
        current_max = 255.0 - (layer_index + 1.0) * gray_value_step
        points = get_layer_points(current_max, point_thresholds[layer_index], prepared_image)

        if len(points) > 1:
            xmax = prepared_image.shape[1]
            ymax = prepared_image.shape[0]


            connected = connections.connect_points(points, max_distance, xmax, ymax)


            for i in range(len(connected)-1):
                start = connected[i]
                end = connected[i+1]
                color = [0]
                stroke_scale=1

                if connections.calc_distance(start,end) < max_distance:
                    # dwg.add(dwg.line(start, end, stroke=svgwrite.rgb(0, 0, 0, '%')))
                    cv2.line(canvas, start, end, color, thickness=stroke_scale, lineType=8, shift=0)
            cv2.imshow("test", canvas)
            cv2.imwrite("./output/result_%04i.png" % layer_index, canvas)
            cv2.waitKey(1)




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
