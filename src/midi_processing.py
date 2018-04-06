# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix

from keras.preprocessing import sequence

from globals import *



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
        self.track_dict = None  # {start_time: [MidiMessage, ...]}
        self.track_C_octaves = Counter()

        self.track = track
        self.ticks_transformer = ticks_transformer
        self.key_sig_transpose = key_sig_transpose
        self.channel = -1



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
            # if it's too long, shorten it
            if this_msg.duration > MAXIMUM_NOTE_LENGTH:
                this_msg.duration = MAXIMUM_NOTE_LENGTH

            if this_msg.start_time not in self.track_dict:
                self.track_dict[this_msg.start_time] = []
            self.track_dict[this_msg.start_time].append(this_msg)

            del self.open_notes[note]

            # save the octave distribution to use to transpose later
            music_note, octave = midi_to_music(this_msg.note)
            # if music_note == "C":
            self.track_C_octaves.update([octave])

        # else:
        #     print("Note off with no start:", note)



    def close_all_notes(self):
        """
        Loops through open notes and closes them all with self.time_now.

        :return: None
        """

        # look for still playing notes and close them if all the messages are done
        for key, value in self.open_notes.copy().items():
            # print("Note has no end:", key)
            # print("       duration:", self.time_now - value.start_time)
            # print("       velocity:", value.velocity)
            # print("Removing from <open_notes>...")

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
            for start_time, note_list in self.track_dict.copy().items():
                for msg in note_list:
                    msg.transpose(octave_transpose)



    def to_dict(self):
        """
        Converts it to a list.

        :return: The messages as a time series in a list.
        """

        self.track_dict = {}

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

                if self.channel == -1:
                    self.channel = msg.channel


        self.close_all_notes()
        # if the track didn't contain any actual notes, only meta
        if not len(self.track_dict.keys()):
            return None

        # transpose it to the most common octave
        # self.transpose_octavewise()

        return self.track_dict




class MidiTrackText(MidiTrack):

    vectorizer = None

    def __init__(self, track, ticks_transformer, key_sig_transpose):
        super().__init__(track, ticks_transformer, key_sig_transpose)


    def to_text(self):
        """
        Converts a track into a list of text representations.

        :return: The notes contained within as a list
        """

        self.to_dict()

        if not self.track_dict:
            return None

        if self.channel != 10:
            result = ["TRACK_START"]
        else:
            result = ["DRUM_TRACK_START"]

        for time, notes in sorted(self.track_dict.items()):
            step = [midi_to_string(msg.note) + ":" + str(bin_note_duration(msg.duration)) for msg in sorted(notes, key=lambda x: x.note)]
            result.append(";".join(step))

        if self.channel != 10:
            result.append("TRACK_END")
        else:
            result.append("DRUM_TRACK_END")


        return result


    def to_sequence(self):
        """
        Converts a track into a list of text representations.

        :return: The notes contained within as a list
        """

        text = self.to_text()

        if not text:
            return None

        result = self.vectorizer.transform(text)

        return result




class MidiTrackNHot(MidiTrack):

    def __init__(self, track, ticks_transformer, key_sig_transpose):
        super().__init__(track, ticks_transformer, key_sig_transpose)



    def to_sequence(self):

        self.to_dict()

        if not self.track_dict:
            return None

        empty = np.zeros(128 + 4 + NOTE_RESOLUTION)

        if self.channel != 10:
            first = empty.copy()
            first[-2] = 1
            result = [first]
        else:
            first = empty.copy()
            first[-4] = 1
            result = [first]

        for time, notes in sorted(self.track_dict.items()):

            n_hot = empty.copy()

            for msg in notes:
                n_hot[msg.note] = 1
                duration = np.amin([int(bin_note_duration(msg.duration) / NOTE_RESOLUTION), NOTE_RESOLUTION])
                n_hot[128 + 4 + duration] = 1

            result.append(n_hot)

        if self.channel != 10:
            last = empty.copy()
            last[-1] = 1
            result.append(last)
        else:
            last = empty.copy()
            last[-3] = 1
            result.append(last)

        return result




class MidiFileBase:

    def __init__(self, filename, meta_df, track_converter):

        self.filename = filename
        self.meta_df = meta_df

        self.mid = mido.MidiFile(self.filename)
        self.key_sig_transpose = self.get_keysig_transpose_interval()
        self.ticks_transformer = TICKS_PER_BEAT / self.mid.ticks_per_beat  # coefficient to convert msg.time

        self.track_converter = track_converter




    def get_keysig_transpose_interval(self):
        """
        Gets the transpose interval for a filename from the meta dataframe.

        :return: The interval to use to transpose this file to the correct key signature.
        """

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



    def to_X(self):
        """
        Converts a mido MidiFile into a list of dictionaries.
        :return: A list of dictionaries.  Each dictionary represents a track.  The dictionaries are in the format
                 {start_time: [tuple(note, duration, velocity), ...]}
        """

        X = []

        for track in self.mid.tracks:

            track_converter = self.track_converter(track, self.ticks_transformer, self.key_sig_transpose)
            track_result = track_converter.to_sequence()

            if track_result == None:
                continue

            if type(track_result) == csr_matrix:
                track_result = np.array(track_result.todense())
            else:
                track_result = np.array(track_result)

            partitions = int(track_result.shape[0] / NUM_STEPS) + 1
            chunks = [track_result[i * NUM_STEPS:(i + 1) * NUM_STEPS] for i in range(partitions)]

            for chunk in chunks:
                chunk = sequence.pad_sequences(chunk.T, maxlen=NUM_STEPS, padding="post").T
                X.append(chunk)

        return X




class MidiFileText(MidiFileBase):


    def __init__(self, filename, meta_df):
        MidiFileBase.__init__(self, filename, meta_df, MidiTrackText)


    def to_text(self):
        """
        Converts a mido MidiFile into a list of dictionaries.
        :return: A list of dictionaries.  Each dictionary represents a track.  The dictionaries are in the format
                 {start_time: [tuple(note, duration, velocity), ...]}
        """

        result = []

        for track in self.mid.tracks:

            track_converter = MidiTrackText(track, self.ticks_transformer, self.key_sig_transpose)
            track_result = track_converter.to_text()

            if track_result:
                result.append(track_result)

        return result




class MidiFileNHot(MidiFileBase):

    def __init__(self, filename, meta_df):
        super().__init__(filename, meta_df, MidiTrackNHot)




if __name__ == "__main__":
    file = "raw_midi/Arcas/Arcas_Fagot_.mid"
    df = pd.read_csv("raw_midi/meta.csv", index_col="filename")
    mid = mido.MidiFile(file)
    t = MidiTrackNHot(mid.tracks[1], 1, 0)
