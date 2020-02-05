import configparser
import argparse
import subprocess
import os




def delete_and_create_output_folder():
    with open(os.devnull, 'wb') as quiet_output:
        subprocess.call(["rm", "-r", "output"])
        subprocess.call(["mkdir", "-p", "output"])

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
    config = configparser.ConfigParser().read(filename)



if __name__ == '__main__':
    main()
