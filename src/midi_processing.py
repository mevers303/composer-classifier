# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido
import pandas as pd
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


class MidiTrack():

    def __init__(self, track, ticks_transformer, key_sig_transpose):
        """
        Initializes the object.

        :param track: A mido.MidiTrack object.
        :param ticks_transformer: The conversion rate for ticks per beat.
        :param key_sig_transpose: The interval to transpose into C/Am
        """

        self.time_now = 0  # absolute time
        self.open_notes = {}  # {msg.note: MidiMessage}
        self.track_list = {}  # {start_time: [MidiMessage, ...]}
        self.track_C_octaves = Counter()

        self.track = track
        self.ticks_transformer = ticks_transformer
        self.key_sig_transpose = key_sig_transpose
    
    
    
    def close_note(self, note):
        """
        Moves an open note from open_notes to track_result.

        :param note: The note to close
        :return: None
        """

        # if it's already playing, take it out of open_notes and add it to our list
        if note in self.open_notes:

            this_msg = self.open_notes[note]
            # IMPORTANT: transpose to correct key signature occurs here
            this_msg.transpose(self.key_sig_transpose)
            this_msg.duration = self.time_now - this_msg.start_time

            if this_msg.start_time not in self.track_list:
                self.track_list[this_msg.start_time] = []
            self.track_list[this_msg.start_time].append(this_msg)

            del self.open_notes[note]

            # save the octave distribution to use to transpose later
            music_note, octave = midi_to_music(this_msg.note)
            # if music_note == "C":
            self.track_C_octaves.update([octave])

        else:
            print("Note off with no start:", note)
    
    
    
    def close_all_notes(self):
        """
        Loops through open notes and closes them all with self.time_now.

        :return: None
        """
        
        # look for still playing notes and close them if all the messages are done
        for key, value in self.open_notes.items():
            print("Note has no end:", key)
            print("       duration:", self.time_now - value.start_time)
            print("       velocity:", value.velocity)
            print("Removing from <open_notes>...")

            self.close_note(key)



    def transpose_octavewise(self):
        """
        Transposes the track to the middle C octave range.

        :return: None
        """

        # transpose it to the correct octave (C4)
        most_common_octave = self.track_C_octaves.most_common(1)[0][0]
        if most_common_octave != 4:

            octave_transpose = (4 - most_common_octave) * 12
            for start_time, note_list in self.track_list.copy().items():
                for msg in note_list:
                    msg.transpose(octave_transpose)



    def to_list(self):
        """
        Converts it to a list.

        :return: The messages as a time series in a list.
        """

        for msg in self.track:

            # msg.time is the time since the last message.  So add this to time to get the current time since the track start
            self.time_now += int(msg.time * self.ticks_transformer)

            if msg.type == "note_off":
                self.close_note(msg.note)
                continue

            if msg.type == 'note_on':

                # if the velocity is 0, that means it is a "note_off" message, close the note and move on
                if msg.velocity == 0:
                    self.close_note(msg.note)
                    continue

                # it shouldn't be open already, but check any way.
                if msg.note in self.open_notes:
                    self.close_note(msg.note)

                # add it to open notes
                self.open_notes[msg.note] = MidiMessage(msg, self.time_now)


        self.close_all_notes()
        # if the track didn't contain any actual notes, only meta
        if not len(self.track_list.keys()):
            return None

        # transpose it to the most common octave
        self.transpose_octavewise()
        return self.track_list
        
        


class MidiVector():

    def __init__(self, filename, meta_df):

        self.filename = filename
        self.meta_df = meta_df

        self.mid = mido.MidiFile(self.filename)
        self.key_sig_transpose = self.get_keysig_transpose_interval()



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
        key_sig = get_key_sig(note_dist)

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


        for track in self.mid.tracks:

            track_converter = MidiTrack(track, ticks_transformer, self.key_sig_transpose)
            track_result = track_converter.to_list()

            if track_result:
                result.append(track_result)

        return result





if __name__ == "__main__":
    df = pd.read_csv("raw_midi/meta.csv", index_col="filename")
    mv = MidiVector("raw_midi/Barrios/Barrios_Estudio_Si_menor.mid", df)
    l = mv.tracks_to_list()
