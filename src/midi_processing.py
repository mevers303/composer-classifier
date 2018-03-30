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

from src.globals import *



def dump_tracks(midi_file):
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")

def dump_msgs(track):
    for i, msg in enumerate(track):
        print(str(i).rjust(4), ": ", msg, sep="")



class MidiArchive():

    def __init__(self, base_dir="midi/"):

        self.base_dir = base_dir
        self.composers = set()

        self.midi_filenames = []
        self.midi_filenames_labels = []
        self.midi_filenames_total = 0
        self.midi_filenames_invalid = []
        self.midi_objects = []
        self.midi_objects_labels = []
        self.midi_filenames_loaded = 0
        self.meta_df = pd.DataFrame(columns=["composer", "type", "ticks_per_beat", "key", "time_n",
                                             "time_d", "time_32nd", "first_note", "first_note_time"])

        self.threads = []
        self.thread_lock = threading.Lock()



    def get_all_filenames(self):
        """Returns a list of files in <dir> and their associated label in y.  Files must be in <dir>/<composer>/*.mid"""

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



    def build_mido_and_meta(self):

        chunk_size = int(self.midi_filenames_total / NUM_THREADS + 1)
        chunkified_filenames = [self.midi_filenames[i:i + chunk_size] for i in
                                range(0, len(self.midi_filenames), chunk_size)]
        chunkified_labels = [self.midi_filenames_labels[i:i + chunk_size] for i in
                             range(0, len(self.midi_filenames_labels), chunk_size)]

        print("Loading midi files with", len(chunkified_filenames), "threads...")
        for filenames, labels in zip(chunkified_filenames, chunkified_labels):
            thread = threading.Thread(target=MidiArchive.build_mido_and_meta_chunk, args=(self, filenames, labels))
            thread.start()
            self.threads.append(thread)

        for thread in self.threads:
            thread.join()


        return self.midi_objects, self.meta_df



    def build_mido_and_meta_chunk(self, filenames, labels):

        for file, composer in zip(filenames, labels):
            self.parse_midi_file(file, composer)

        with self.thread_lock:
            print("\nThread finished!")



    def parse_midi_file(self, file, composer):

        key_sig = time_n = time_d = time_32nd = first_note = first_note_time = None

        try:
            mid = mido.MidiFile(file)

            for msg in mid:

                # break if we already know the key signature, time signature, and first note
                if key_sig and time_n and first_note:
                    break

                if msg.type == "key_signature":
                    if not key_sig:
                        key_sig = msg.key
                elif msg.type == "time_signature":
                    if not time_n:
                        time_n = msg.numerator
                        time_d = msg.denominator
                        time_32nd = msg.notated_32nd_notes_per_beat
                elif msg.type == "note_on":
                    if not first_note:
                        first_note = msg.note
                        first_note_time = msg.time

            with self.thread_lock:
                self.meta_df.loc[file] = [composer, mid.type, mid.ticks_per_beat, key_sig, time_n, time_d, time_32nd,
                                          first_note, first_note_time]
                self.midi_objects.append(mid)
                self.midi_objects_labels.append(composer)
                self.midi_filenames_loaded += 1
                progress_bar(self.midi_filenames_loaded, self.midi_filenames_total)


        except:
            with self.thread_lock:
                print("\nERROR -> Skipping invalid file:", file)
                self.midi_filenames_invalid.append(file)
                self.midi_filenames_loaded += 1
                progress_bar(self.midi_filenames_loaded, self.midi_filenames_total)




def track_to_list(track):
    """We need to loop """

    time_now = 0  # start of the track
    open_notes = {}  # key = note, value = tuple(index_in_result, velocity, start_time)
    result = []  # list of tuple(note, velocity, duration)

    def close_note(note):
        """Inner function to move a note from open_notes to result."""

        # if it's already playing, take it out of open_notes and add it to our list
        if note in open_notes:
            open_note_i, velocity, start_time = open_notes[note]
            duration = time_now - start_time
            result[open_note_i] = (note, velocity, duration)  # change the duration for this note in result
            del open_notes[note]

        else:
            print("Note off with no start:", msg.note)


    for msg in track:

        #  msg.time is the time since the last message.  So add this to time to get the current time since the track start
        time_now += msg.time

        if msg.type == 'note_on':

            # if the velocity is 0, that means it is a "key up" message, close the note and move on
            if msg.velocity == 0:
                close_note(msg.note)
                continue

            # it shouldn't be open already, but check any way.
            if msg.note in open_notes:
                close_note(msg.note)

            # add it to open notes and result (result will be modified later)
            open_notes[msg.note] = (len(result), msg.velocity, time_now)  # len(result is the index of what's about to be added
            result.append((msg.note, msg.velocity, time_now))


    # look for still playing notes and close them if all the messages are done
    for key, value in open_notes.items():

        print("Note has no end:", key)
        print("       velocity:", value[1])
        print("       duration:", time_now - value[2])
        print("Removing from <open_notes>...")

        close_note(key)


    return result



if __name__ == "__main__":

    archive = MidiArchive()
    archive.get_all_filenames()
    midis, df = archive.build_mido_and_meta()

    df.to_csv("midi/100_per_composer.csv")
    pickle.dump(midis, "midi/100_per_composer.pkl")
