# Mark Evers
# Created: 3/27/2018
# midi_file.py
# Functions for processing MIDI files


class MidiMessage:

    def __init__(self, msg, start_time):
        self.note = msg.note
        self.velocity = msg.velocity
        self.duration = None
        self.channel = msg.channel
        self.start_time = start_time

    def transpose(self, interval):
        if self.channel != 9:
            self.note += interval
