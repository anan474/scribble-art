import configparser
import argparse
import subprocess
import os
import cv2
import random
import numpy as np
import math
import sys
import connections

def get_empty_white_canvas(size_x=1920, size_y=1080):
    """
    This returns the array on which will be drawn.
    """
    img = np.array([255], dtype=np.uint8) * np.ones((size_y,size_x,3), dtype=np.uint8)
    return img


def get_prepared_image(source_image, scale_factor):
    """
    The original image must be resized and turned into a
    grayscale image.
    """
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    new_width = int(scale_factor * gray_image.shape[1])
    new_height = int(scale_factor * gray_image.shape[0])
    new_size = (new_width, new_height)
    resized_image = cv2.resize(gray_image, new_size)
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


    max_distance = float(config["DRAWING"]["max_line_length_factor"]) * min(prepared_image.shape[1], prepared_image.shape[0])

    lines = []
    for layer_index in range(no_of_layers):
        text = "\rCreate layer {:4d} / {:d}".format(layer_index, no_of_layers-1)
        sys.stdout.write(text)
        sys.stdout.flush()
        current_max = 255.0 - (layer_index + 1.0) * gray_value_step
        points = get_layer_points(current_max, point_thresholds[layer_index], prepared_image)

        if len(points) > 1:
            xmax = prepared_image.shape[1]
            ymax = prepared_image.shape[0]
            connected = connections.connect_points(points, max_distance, xmax, ymax)


            for i in range(len(connected)-1):
                start = connected[i]
                end = connected[i+1]
                if connections.calc_distance(start,end) < max_distance:
                    lines.append([start,end])
    print("")

    if bool(config["INPUT_OUTPUT"]["create_video"]):
        video_parameters = config["VIDEO_PARAMETERS"]
        create_video(lines, video_parameters, prepared_image.shape)
    if bool(config["INPUT_OUTPUT"]["create_png"]):
        canvas = create_final_canvas(lines, prepared_image.shape)



def create_final_canvas(lines, shape):
    canvas = get_empty_white_canvas(shape[1], shape[0])
    for line in lines:
        start = line[0]
        end = line[1]
        stroke_scale = 1
        color = [0,0,0]
        cv2.line(canvas, start, end, color, thickness=stroke_scale, lineType=8, shift=0)
    return canvas

def create_video(lines, video_parameters, shape):
    with open(os.devnull, 'wb') as quiet_output:
        subprocess.call(["mkdir", "output/frames"])

    drawing_duration = float(video_parameters["drawing_duration"])
    fps = float(video_parameters["fps"])
    no_of_frames = int(drawing_duration * fps)
    no_of_lines_per_frame = int(len(lines) / (drawing_duration * fps)) + 1
    no_of_lines_per_second = int(len(lines) / drawing_duration)
    frames = []
    for i in range(no_of_frames):
        canvas = get_empty_white_canvas(shape[1], shape[0])
        text = "\rCreate frame {:7d} / {:d}".format(i, no_of_frames-1)
        sys.stdout.write(text)
        sys.stdout.flush()
        line_index_a = i * no_of_lines_per_frame
        line_index_b = min(line_index_a + no_of_lines_per_frame, len(lines))
        for line_index, line in enumerate(lines[0:line_index_b]):
            start = line[0]
            end = line[1]
            seconds_lines_remain_colored = float(video_parameters["seconds_lines_remain_colored"])
            if line_index > line_index_b - seconds_lines_remain_colored * no_of_lines_per_second:
                color = [int(c) for c in video_parameters["active_line_color"].split(",")]
                stroke_scale = 2
            else:
                stroke_scale = 1
                color = [0,0,0]
            cv2.line(canvas, start, end, color, thickness=stroke_scale, lineType=8, shift=0)
        frames.append(canvas)
    print("")

    # final frame
    canvas = get_empty_white_canvas(shape[1], shape[0])
    for line in lines:
        start = line[0]
        end = line[1]
        stroke_scale = 1
        color = [0,0,0]
        cv2.line(canvas, start, end, color, thickness=stroke_scale, lineType=8, shift=0)
    duration_of_final_image = float(video_parameters["duration_of_final_image"])
    for i in range(int(fps) * 5):
        frames.append(canvas)

    size = (shape[1],shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./output/final.avi', fourcc, fps, size)

    for f in frames:
        out.write(f)
    out.release()


def delete_and_create_output_folder():
    """
    Each time the program is run, the previous output
    folder shall be deleted.
    """
    with open(os.devnull, 'wb') as quiet_output:
        subprocess.call(["rm", "-r", "output"])
        subprocess.call(["mkdir", "output"])


def get_config(filename):
    """
    All settings are stored in an external text file.
    """
    config = configparser.ConfigParser()
    config.read(filename)
    return config

def set_seeds_of_rngs(seed):
    """
    You should be able to set the seeds of
    the employed random number generators
    if you want to reproduce an image.
    """
    random.seed(seed)
    np.random.seed(seed)

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
    set_seeds_of_rngs(int(config["DRAWING"]["random_seed"]))
    create_scribble_art(config)


if __name__ == '__main__':
    main()
