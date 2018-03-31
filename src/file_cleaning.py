# Mark Evers
# Created: 3/28/2018
# midi_processing.py
# Functions for managing MIDI files

import os
import shutil
import string

from src.globals import *



def move_to_folders(source_folder, dest_folder, split_char="_", skip_scarce=False):
    """Moves files in source_folder to <composer>/*.mid.  They must have format <composer>_*.mid.Arguments must have trailing slash."""

    composers = {}

    if not dest_folder.endswith("/"):
        dest_folder += "/"
    if not source_folder.endswith("/"):
        source_folder += "/"


    for file in os.listdir(source_folder):

        if not (file.endswith(".mid") or file.endswith(".midi")):
            continue

        composer = file.split(split_char)[0]
        if not composer in composers:
            composers[composer] = []

        composers[composer].append(file)


    for composer, files in composers.items():

        if skip_scarce and len(files) < MINIMUM_WORKS:
            continue

        composer_dir = dest_folder + composer + "/"
        composer_dir = composer_dir.capitalize()
        if not os.path.exists(composer_dir):
            os.makedirs(composer_dir)

        for file in files:
            shutil.copyfile(source_folder + file, composer_dir + file)


    return composers



def capitalize_folders(dir):
    """Capitalizes the first letter of all folders in dir."""

    if not dir.endswith("/"):
        dir += "/"

    for file in os.listdir(dir):

        path = dir + file

        if not os.path.isdir(path):
            continue

        if file[0] in string.ascii_uppercase:
            continue

        new_name = dir + file.capitalize()

        if not os.path.exists(new_name):
            # print("Rename {} -> {}".format(dir + file, new_name))
            os.rename(dir + file, new_name)
        else:
            print("Directory already exists:", new_name)



def remove_format_0(dir="raw_midi/"):

    for root, dirs, files in os.walk(dir):

        for file in files:
            if file.find("_format0") > -1:
                orig = file.replace("_format0", "")
                if os.path.exists(os.path.join(root, orig)):
                    print("Removing", os.path.join(root, file))
                    os.remove(os.path.join(root, file))


if __name__ == "__main__":
    remove_format_0()
