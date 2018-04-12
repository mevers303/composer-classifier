# Mark Evers
# Created: 3/27/2018
# midi_file.py
# Functions for processing MIDI files

import mido
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from keras.preprocessing import sequence

from midi_handlers.midi_track import MidiTrackText, MidiTrackNHot, MidiTrackNHotTimeSeries
from globals import *



class MidiFileBase:

    def __init__(self, filename, note_dist, track_converter):

        self.filename = filename
        self.note_dist = note_dist

        self.mid = mido.MidiFile(self.filename)
        self.key_sig_transpose = self.get_keysig_transpose_interval()
        self.ticks_transformer = TICKS_PER_BEAT / self.mid.ticks_per_beat  # coefficient to convert msg.time

        self.track_converter = track_converter




    def get_keysig_transpose_interval(self):
        """
        Gets the transpose interval for a filename from the meta dataframe.

        :return: The interval to use to transpose this file to the correct key signature.
        """

        # get the key signature
        key_sig = get_key_sig(self.note_dist)

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

            try:
                if track_result == None:
                    continue
            except ValueError:
                # it's a numpy array and doesn't like being compared to None
                pass

            if type(track_result) == csr_matrix:
                track_result = np.array(track_result.todense(), dtype=np.byte)
            elif type(track_result) != np.ndarray:
                track_result = np.array(track_result, dtype=np.byte)


            partitions = int(track_result.shape[0] / NUM_STEPS) + 1
            chunks = []
            for i in range(partitions):

                if i:

                    #  take an overlapping chunk from the step before
                    pre_chunk = track_result[(i * NUM_STEPS) - int(NUM_STEPS / 2):((i + 1) * NUM_STEPS) - int(NUM_STEPS / 2)]
                    if pre_chunk.shape[0] < NUM_STEPS:
                        chunks.append(sequence.pad_sequences(pre_chunk.T, maxlen=NUM_STEPS, padding="post").T)
                    else:
                        chunks.append(pre_chunk)

                chunk = track_result[i * NUM_STEPS:(i + 1) * NUM_STEPS]
                if not chunk.shape[0]:
                    continue
                elif chunk.shape[0] < NUM_STEPS:
                    chunks.append(sequence.pad_sequences(chunk.T, maxlen=NUM_STEPS, padding="post").T)
                else:
                    chunks.append(chunk)

            X.extend(chunks)

        return X




class MidiFileText(MidiFileBase):


    def __init__(self, filename, note_dist):
        MidiFileBase.__init__(self, filename, note_dist, MidiTrackText)


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

    def __init__(self, filename, note_dist):
        super().__init__(filename, note_dist, MidiTrackNHot)



class MidiFileNHotTimeSeries(MidiFileBase):

    def __init__(self, filename, note_dist):
        super().__init__(filename, note_dist, MidiTrackNHotTimeSeries)




if __name__ == "__main__":
    file = "midi/classical/Arndt/Nola, Novelty piano solo.mid"
    df = pd.read_csv("midi/classical/meta.csv", index_col="filename")
    mid = mido.MidiFile(file)
    t = MidiFileNHotTimeSeries(file, df.loc[file][MUSIC_NOTES])
    x = t.to_X()
    print(x)
    pass
