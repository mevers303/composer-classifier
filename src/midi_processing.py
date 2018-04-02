# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido
import os
import sys
import pandas as pd
import random
import threading
import time
import pickle
import json
import numpy as np

from src.globals import *

#               0     1    2    3     4    5    6     7    8     9    10    11
music_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
key_signatures = [[ 0,  2,  4,  5,  7,  9, 11],
                  [ 1,  3,  5,  6,  8, 10,  0],
                  [ 2,  4,  6,  7,  9, 11,  1],
                  [ 3,  5,  7,  8, 10,  0,  2],
                  [ 4,  6,  8,  9, 11,  1,  3],
                  [ 5,  7,  9, 10,  0,  2,  4],
                  [ 6,  8, 10, 11,  1,  3,  5],
                  [ 7,  9, 11,  0,  2,  4,  6],
                  [ 8, 10,  0,  1,  3,  5,  7],
                  [ 9, 11,  1,  2,  4,  6,  8],
                  [10,  0,  2,  3,  5,  7,  9],
                  [11,  1,  3,  4,  6,  8, 10]]


def dump_tracks(midi_file):
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")

def dump_msgs(track):
    for i, msg in enumerate(track):
        print(str(i).rjust(4), ": ", msg, sep="")

def midi_to_music(midi_note):

    music_note = music_notes[midi_note % 12]
    octave = int(midi_note / 12) - 1

    return music_note, octave

def midi_to_string(midi_note):
    note = midi_to_music(midi_note)
    return note[0] + str(note[1])





class MidiArchiveMeta():
    """
    Class for building the metadata for a MIDI archive.
    """

    def __init__(self, base_dir="raw_midi/"):
        """
        :param base_dir: The base directory where the MIDI files are contained in composer subfolders.
        """

        self.base_dir = base_dir
        self.composers = set()

        self.midi_filenames = []
        self.midi_filenames_labels = []
        self.midi_filenames_total = 0
        self.midi_filenames_invalid = []
        self.midi_filenames_parsed = 0

        columns = ["composer", "type", "tracks", "ticks_per_beat", "key", "first_time_n", "first_time_d", "first_time_32nd", "time_clocks_per_click", "first_note", "first_note_time", "has_note_off", "has_key_change"]
        columns.extend(music_notes)
        columns.extend(["midi_" + str(i) for i in range(128)])
        self.meta_df = pd.DataFrame(columns=columns)
        self.meta_df.index.name = "filename"

        self.threads = []
        self.thread_lock = None
        self.stop_threads = False

        self.key_sigs = set()
        self.time_sigs = set()



    def get_all_filenames(self):
        """
        Returns a list of files in <self.base_dir> and their associated label in y.  Files must be in
        <dir>/<composer>/*.mid"
        :return: None
        """""

        self.midi_filenames = []
        self.midi_filenames_labels = []
        self.composers = set()


        for composer in os.listdir(self.base_dir):


            composer_files = []
            for root, dirs, files in os.walk(os.path.join(self.base_dir, composer)):

                for file in files:

                    full_path = os.path.join(root, file)
                    if not (file.lower().endswith(".mid") or file.lower().endswith(".midi")):
                        print("Unknown file:", full_path)
                        continue

                    composer_files.append(full_path)


            composer_works = len(composer_files)
            if composer_works < MINIMUM_WORKS:
                # print("Not enough works for {}, ({})".format(composer, composer_works))
                continue
            if composer_works > MAXIMUM_WORKS:
                composer_files = random.sample(composer_files, MAXIMUM_WORKS)
                composer_works = len(composer_files)


            self.midi_filenames.extend(composer_files)
            self.midi_filenames_labels.extend([composer] * composer_works)
            self.composers.add(composer)
            # print("Added {} ({})".format(composer, composer_works))


        self.midi_filenames_total = len(self.midi_filenames)
        print("Found", self.midi_filenames_total, "files from", len(self.composers), "composers!")

        return self.midi_filenames, self.midi_filenames_labels



    def build_meta(self):
        """
        Builds the meta data pandas dataframe
        :return: None
        """

        chunk_size = int(self.midi_filenames_total / NUM_THREADS + 1)
        chunkified_filenames = [self.midi_filenames[i:i + chunk_size] for i in
                                range(0, len(self.midi_filenames), chunk_size)]
        chunkified_labels = [self.midi_filenames_labels[i:i + chunk_size] for i in
                             range(0, len(self.midi_filenames_labels), chunk_size)]


        print("Loading midi files with", len(chunkified_filenames), "threads...")
        self.thread_lock = threading.Lock()
        self.stop_threads = False

        for filenames, labels in zip(chunkified_filenames, chunkified_labels):
            thread = threading.Thread(target=MidiArchiveMeta.build_meta_chunk,
                                      args=(self, filenames, labels))
            thread.start()
            self.threads.append(thread)

        try:
            for thread in self.threads:
                thread.join()

        except KeyboardInterrupt:
            self.stop_threads = True
            raise KeyboardInterrupt

        self.threads = []
        self.thread_lock = None


        return self.meta_df



    def build_meta_chunk(self, filenames, labels):
        """
        :param filenames: list of paths to MIDI files
        :param labels: list of labels (composers) for the MIDI files
        :return: None

        Gets the metadata for a list of files.  This exists as a chunk to work with threading.
        """

        for file, composer in zip(filenames, labels):
            if self.stop_threads:
                break
            self.parse_midi_meta(file, composer)

        # with self.thread_lock:
        #     print("\nThread finished!")



    def parse_midi_meta(self, file, composer):
        """
        :param file: path to a MIDI file
        :param composer: the label (composer) for this file
        :return: None

        Adds a MIDI file's metadata to the meta_df pandas dataframe.
        """

        key_sig = time_n = time_d = time_32nd = time_clocks_per_click = first_note = first_note_time = None
        has_note_off = has_key_change = False
        music_notes_before_key_change = np.zeros((12,))
        midi_notes_before_key_change = np.zeros((128,))

        time_now = 0
        last_key_change_time = 0
        last_key = None

        try:
            mid = mido.MidiFile(file)

            for msg in mid:

                time_now += msg.time


                if msg.type == "key_signature":
                    if msg.key == last_key:
                        continue

                    self.key_sigs.add(msg.key)
                    # if 0 < time_now - last_key_change_time < 5:
                    #     print("\nVery short key signature change ({}s) -> {}".format(time_now - last_key_change_time, mid.filename))

                    if not key_sig or not music_notes_before_key_change.sum():
                        key_sig = msg.key
                    elif time_now - last_key_change_time != 0:  # if the time since the last key change is zero
                        has_key_change = True

                    last_key_change_time = time_now
                    last_key = msg.key


                elif msg.type == "time_signature":
                    self.time_sigs.add("{}/{}/{}/{}".format(msg.numerator, msg.denominator, msg.notated_32nd_notes_per_beat, msg.clocks_per_click))
                    if not time_n:
                        time_n = msg.numerator
                        time_d = msg.denominator
                        time_32nd = msg.notated_32nd_notes_per_beat
                        time_clocks_per_click = msg.clocks_per_click


                elif msg.type == "note_on":
                    if not msg.velocity:
                        continue
                    if not first_note:
                        first_note = midi_to_music(msg.note)[0]
                        first_note_time = msg.time
                    if not has_key_change:
                        midi_notes_before_key_change[msg.note] += 1
                        music_notes_before_key_change[msg.note % 12] += 1


                elif msg.type == "note_off":
                    if not has_note_off:
                        has_note_off = True


            with self.thread_lock:

                values = [composer, mid.type, len(mid.tracks), mid.ticks_per_beat, key_sig, time_n, time_d, time_32nd, time_clocks_per_click, first_note, first_note_time, has_note_off, has_key_change]
                values.extend(music_notes_before_key_change)
                values.extend(midi_notes_before_key_change)
                self.meta_df.loc[file] = values

                self.midi_filenames_parsed += 1
                progress_bar(self.midi_filenames_parsed, self.midi_filenames_total)


        except KeyboardInterrupt:
            # do nothing
            raise KeyboardInterrupt

        except:
            with self.thread_lock:
                print("\nERROR -> Skipping invalid file:", file)
                self.midi_filenames_invalid.append(file)
                self.midi_filenames_parsed += 1
                progress_bar(self.midi_filenames_parsed, self.midi_filenames_total)





class MidiArchiveVector():

    def __init__(self, meta_df):

        self.meta_df = meta_df


    def tracks_to_list(self, mid):
        """
        Converts a mido MidiFile into a list of dictionaries.
        :param mid: A mido.MidiFile object
        :return: A list of dictionaries.  Each dictionary represents a track.  The dictionaries are in the format
                 {start_time: [tuple(note, duration, velocity), ...]}
        """

        ticks_transformer = TICKS_PER_BEAT / mid.ticks_per_beat  # coefficient to convert msg.time
        result = []

        def close_note(note):
            """
            Inner function to move a note from open_notes to track_result.
            :param note: The note to close
            :return: None
            """

            # if it's already playing, take it out of open_notes and add it to our list
            if note in open_notes:

                start_time, velocity = open_notes[note]
                duration = time_now - start_time

                if start_time not in track_result:
                    track_result[start_time] = []
                track_result[start_time].append((note, duration, velocity))

                del open_notes[note]

            else:
                print("Note off with no start:", msg.note)



        for track in mid.tracks:

            time_now = 0  # absolute time
            open_notes = {}  # {msg.note: tuple(start_time, velocity)}
            track_result = {}  # {start_time: [tuple(note, duration, velocity), ...]}

            for msg in track:

                # msg.time is the time since the last message.  So add this to time to get the current time since the track start
                time_now += int(msg.time * ticks_transformer)

                if msg.type == "note_off":
                    close_note(msg.note)
                    continue

                if msg.type == 'note_on':

                    # if the velocity is 0, that means it is a "note_off" message, close the note and move on
                    if msg.velocity == 0:
                        close_note(msg.note)
                        continue

                    # it shouldn't be open already, but check any way.
                    if msg.note in open_notes:
                        close_note(msg.note)

                    # add it to open notes
                    open_notes[msg.note] = (time_now, msg.velocity)


            # look for still playing notes and close them if all the messages are done
            for key, value in open_notes.items():

                print("Note has no end:", key)
                print("       velocity:", value[1])
                print("       duration:", time_now - value[2])
                print("Removing from <open_notes>...")

                close_note(key)


            if len(track_result.keys()):
                result.append(track_result)



        return result






def build_all_meta(dir="raw_midi"):

    global MAXIMUM_WORKS
    global MINIMUM_WORKS
    MAXIMUM_WORKS = 10000
    MINIMUM_WORKS = 1

    archive = MidiArchiveMeta(dir)
    archive.get_all_filenames()
    df = archive.build_meta()

    for file in archive.midi_filenames_invalid:
        archive.midi_filenames.remove(file)
        os.remove(file)


    print("Saving meta csv...")
    df.to_csv(os.path.join(dir, "meta.csv"))
    print("Meta CSV file saved!")

    info = {"key_sigs": list(archive.key_sigs), "time_sigs": list(archive.time_sigs)}
    with open(os.path.join(dir, "info.json"), "w") as f:
        json.dump(info, f)
    print("JSON file saved!")



if __name__ == "__main__":
    build_all_meta("/media/mark/Windows/raw_midi")
    pass
