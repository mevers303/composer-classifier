# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido
import pandas as pd
import numpy as np
from collections import Counter

from src.globals import *



class MidiMessage():

    def __init__(self, msg, start_time):
        self.note = msg.note
        self.velocity = msg.velocity
        self.duration = None
        self.channel = msg.channel
        self.start_time = start_time

    def transpose(self, interval):
        if self.channel != 10:
            self.note += interval



class MidiVector():

    def __init__(self, filename, meta_df):

        self.filename = filename
        self.meta_df = meta_df

        self.mid = mido.MidiFile(self.filename)
        self.key_sig_transpose = self.get_keysig_transpose_interval()



    def get_key_sig(self, note_dist):
        """
        Uses the note distribution in the meta dataframe to determine a filename's key signature
        :param note_dist: Distribution of notes as per music_notes
        :return: The index of key_signatures that is the best match.
        """

        top_notes = set(np.argsort(note_dist)[::-1][:7])


        best_match = -1
        best_match_set_dif_len = 100

        for i in range(len(KEY_SIGNATURES)):

            # find number of uncommon notes
            set_dif_len = len(set(KEY_SIGNATURES[i]) - top_notes)

            # if this one is better than the last, save it
            if set_dif_len < best_match_set_dif_len:
                best_match = i
                best_match_set_dif_len = set_dif_len

                # if there are 0 uncommon notes, it is our match!
                if not best_match_set_dif_len:
                    break


        return best_match



    def get_keysig_transpose_interval(self):
        """
        Gets the transpose interval for a filename from the meta dataframe.
        :param filename: Path to a midi file
        :return: The interval to use to transpose this file to the correct key signature.
        """

        transpose_interval = 0

        row = self.meta_df.loc[self.filename]

        # get the key signature
        note_dist = row[MUSIC_NOTES].values.astype(int)
        key_sig = self.get_key_sig(note_dist)

        # first transpose based on key signature
        if key_sig < 6:
            transpose_interval = -key_sig
        else:
            transpose_interval = 12 - key_sig


        return transpose_interval



    def tracks_to_list(self):
        """
        Converts a mido MidiFile into a list of dictionaries.
        :param mid: A mido.MidiFile object
        :return: A list of dictionaries.  Each dictionary represents a track.  The dictionaries are in the format
                 {start_time: [tuple(note, duration, velocity), ...]}
        """

        ticks_transformer = TICKS_PER_BEAT / self.mid.ticks_per_beat  # coefficient to convert msg.time
        result = []


        def close_note(note):
            """
            Inner function to move a note from open_notes to track_result.
            :param note: The note to close
            :return: None
            """

            # if it's already playing, take it out of open_notes and add it to our list
            if note in open_notes:

                this_msg = open_notes[note]
                # IMPORTANT: transpose to correct key signature occurs here
                this_msg.transpose(self.key_sig_transpose)
                this_msg.duration = time_now - this_msg.start_time

                if this_msg.start_time not in track_result:
                    track_result[this_msg.start_time] = []
                track_result[this_msg.start_time].append(this_msg)

                del open_notes[note]

                # save the octave distribution to use to transpose later
                music_note, octave = midi_to_music(this_msg.note)
                # if music_note == "C":
                track_C_octaves.update([octave])

            else:
                print("Note off with no start:", note)



        for track in self.mid.tracks:

            time_now = 0  # absolute time
            open_notes = {}  # {msg.note: MidiMessage}
            track_result = {}  # {start_time: [MidiMessage, ...]}
            track_C_octaves = Counter()


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
                    open_notes[msg.note] = MidiMessage(msg, time_now)


            # look for still playing notes and close them if all the messages are done
            for key, value in open_notes.items():

                print("Note has no end:", key)
                print("       duration:", time_now - value.start_time)
                print("       velocity:", value.velocity)
                print("Removing from <open_notes>...")

                close_note(key)


            if not len(track_result.keys()):
                continue


            # transpose it to the correct octave (C4)
            most_common_octave = track_C_octaves.most_common(1)[0][0]
            if most_common_octave != 4:

                octave_transpose = (4 - most_common_octave) * 12
                for start_time, note_list in track_result.copy().items():
                    for msg in note_list:
                        msg.transpose(octave_transpose)


            result.append(track_result)



        return result





if __name__ == "__main__":
    df = pd.read_csv("/media/mark/Windows/raw_midi/meta.csv", index_col="filename")
    mv = MidiVector("/media/mark/Windows/raw_midi/Barrios/Barrios_Estudio_Si_menor.mid", df)
    l = mv.tracks_to_list()
