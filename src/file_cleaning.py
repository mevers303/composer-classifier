# Mark Evers
# Created: 3/28/2018
# midi_processing.py
# Functions for managing MIDI files

import os
import shutil


MIN_WORKS = 50



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

        if skip_scarce and len(files) < MIN_WORKS:
            continue

        composer_dir = dest_folder + composer + "/"
        if not os.path.exists(composer_dir):
            os.makedirs(composer_dir)

        for file in files:
            shutil.copyfile(source_folder + file, composer_dir + file)


    return composers


if __name__ == "__main__":

    folder = "/home/mark/Documents/midi/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Classical_Violin_theviolinsite.com_MIDIRip"

    composers = move_to_folders(folder, folder, "-")
