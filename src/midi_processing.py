# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido
import os
import sys
import pandas as pd
import random



from src.globals import *



def dump_tracks(midi_file):
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")



def dump_msgs(track):
    for i, msg in enumerate(track):
        print(str(i).rjust(4), ": ", msg, sep="")



def get_all_filenames(dir = "midi/"):
    """Returns a list of files in <dir> and their associated label in y.  Files must be in <dir>/<composer>/*.mid"""

    midi_files = []
    y = []
    composers = set()

    for composer in os.listdir(dir):

        composer_files = []

        for root, dirs, files in os.walk(os.path.join(dir, composer)):

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

        midi_files.extend(composer_files)
        y.extend([composer] * composer_works)
        composers.add(composer)
        # print("Added {} ({})".format(composer, composer_works))


    print("Found", len(midi_files), "files from", len(composers), "composers!")
    return midi_files, y



def build_mido_and_meta(filenames, y):


    df = pd.DataFrame(index=["filename"], columns=["composer", "type", "ticks_per_beat", "key", "time_n", "time_d", "time_32nd", "first_note", "first_note_time"])
    midi_objects = []

    n = len(filenames)
    i = 0


    for file, composer in zip(filenames, y):
        i += 1

        try:
            mid = mido.MidiFile(file)
            key_sig = time_n = time_d = time_32nd = first_note = first_note_time = None

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

            df.loc[file] = [composer, mid.type, mid.ticks_per_beat, key_sig, time_n, time_d, time_32nd, first_note, first_note_time]
            midi_objects.append(mid)


        except:
            print("\nERROR -> Skipping invalid file:", file)
            continue

        progress_bar(i, n)


    return midi_objects, df



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

    files, y = get_all_filenames()
    midi_files, df = build_mido_and_meta(files, y)
