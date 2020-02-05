import configparser
import argparse
import subprocess






def main():
    subprocess.call(["rm", "-r", "output"])
    subprocess.call(["mkdir", "-p", "output"])
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



if __name__ == '__main__':
    main()
