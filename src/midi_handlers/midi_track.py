# Mark Evers
# Created: 3/27/2018
# midi_file.py
# Functions for processing MIDI files

import numpy as np
from collections import Counter

from midi_handlers.midi_message import MidiMessage
from globals import *



class MidiTrack:

    def __init__(self, track, ticks_transformer, key_sig_transpose):
        """
        Initializes the object.

        :param track: A mido.MidiTrack object.
        :param ticks_transformer: The conversion rate for ticks per beat.
        :param key_sig_transpose: The interval to transpose into C/Am
        """

        self.time_now = None  # absolute time
        self.open_notes = {}  # {msg.note: MidiMessage}
        self.track_dict = None  # {start_time: [MidiMessage, ...]}
        self.track_C_octaves = Counter()
        self.program = 0

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
            # bin it to the correct duration
            this_msg.duration = bin_note_duration(self.time_now - this_msg.start_time)

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
            # we haven't seen a note_on yet, it's just some meta message at the beginning.  we want to make the first note at time = 0
            if self.time_now != None:
                self.time_now += int(msg.time * self.ticks_transformer)

            # get the instrument
            if msg.type == "program_change":
                self.program = msg.program

            if msg.type == "note_off":
                self.close_note(msg.note)
                continue

            if msg.type == 'note_on':

                if self.time_now == None:
                    self.time_now = 0

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
        self.transpose_octavewise()

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

        if self.channel != 9:
            result = [str(self.program) + "_TRACK_START"]
        else:
            result = ["DRUM_TRACK_START"]

        for time, notes in sorted(self.track_dict.items()):
            step = [midi_to_string(msg.note) + ":" + str(msg.duration) for msg in sorted(notes, key=lambda x: x.note)]
            result.append(";".join(step))

        if self.channel != 9:
            result.append("TRACK_END")
        else:
            result.append(str(self.program) + "_TRACK_END")

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

        track_on_i = 128 + len(DURATION_BINS)
        track_off_i = track_on_i + 1
        drum_track_on_i = track_off_i + 1
        drum_track_off_i = drum_track_on_i + 1

        empty = np.zeros(drum_track_off_i + 1, dtype=np.byte)

        if self.channel != 9:
            first = empty.copy()
            first[track_on_i] = 1
            result = [first]
        else:
            first = empty.copy()
            first[drum_track_on_i] = 1
            result = [first]

        for time, notes in sorted(self.track_dict.items()):

            n_hot = empty.copy()

            for msg in notes:
                n_hot[msg.note] = 1
                duration = DURATION_BINS.index(msg.duration)
                n_hot[128 + duration] = 1

            result.append(n_hot)

        if self.channel != 9:
            last = empty.copy()
            last[track_off_i] = 1
            result.append(last)
        else:
            last = empty.copy()
            last[drum_track_off_i] = 1
            result.append(last)

        return result



class MidiTrackNHotTimeSeries(MidiTrack):


    def __init__(self, track, ticks_transformer, key_sig_transpose):
        super().__init__(track, ticks_transformer, key_sig_transpose)



    @staticmethod
    def bin_timestamp(time):

        bin = int(time / MINIMUM_TIMESERIES_STEP)
        remainder = time % MINIMUM_TIMESERIES_STEP

        if remainder > MINIMUM_TIMESERIES_STEP / 2:
            bin += 1

        return bin + 1  # add 1 because the first is the special track_on note


    def to_sequence(self):

        self.to_dict()

        if not self.track_dict:
            return None


        track_on_i = 128 + len(DURATION_BINS)
        track_off_i = track_on_i + 1
        drum_track_on_i = track_off_i + 1
        drum_track_off_i = drum_track_on_i + 1
        time_len = self.bin_timestamp(max(self.track_dict.keys())) + 1 + 2  # add 1 because we want the length, not the index.  add 2 because of track_on and track_off notes

        result = np.zeros(shape=(time_len, drum_track_off_i + 1), dtype=np.byte)


        if self.channel != 9:
            result[0, track_on_i] = 1
            result[-1, track_off_i] = 1
        else:
            result[0, drum_track_on_i] = 1
            result[-1, drum_track_off_i] = 1


        for time, notes in sorted(self.track_dict.items()):

            time_i = self.bin_timestamp(time)

            for msg in notes:
                result[time_i, msg.note] = 1
                duration = DURATION_BINS.index(msg.duration)
                result[time_i, 128 + duration] = 1


        return result
