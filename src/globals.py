# Mark Evers
# Created: 3/30/2018
# globals.py
# Global variables and functions

from sys import stdout



####################### OPTIONS #########################
# How many pieces must a composer have for us to consider them?
MINIMUM_WORKS = 100
# How many pieces will we use from each composer?
MAXIMUM_WORKS = 200

# How many threads to use when parsing the MIDI archive?
NUM_THREADS = 3
# How many ticks per beat should each track be converted to?
TICKS_PER_BEAT = 1024



####################### CONSTANTS #######################
#               0     1    2    3     4    5    6     7    8     9    10    11
MUSIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_SIGNATURES = [[0, 2, 4, 5, 7, 9, 11],  # C
                  [ 1,  3,  5,  6,  8, 10,  0],  # C#
                  [ 2,  4,  6,  7,  9, 11,  1],  # D
                  [ 3,  5,  7,  8, 10,  0,  2],  # D#
                  [ 4,  6,  8,  9, 11,  1,  3],  # E
                  [ 5,  7,  9, 10,  0,  2,  4],  # F
                  [ 6,  8, 10, 11,  1,  3,  5],  # F#
                  [ 7,  9, 11,  0,  2,  4,  6],  # G
                  [ 8, 10,  0,  1,  3,  5,  7],  # G#
                  [ 9, 11,  1,  2,  4,  6,  8],  # A
                  [10,  0,  2,  3,  5,  7,  9],  # A#
                  [11,  1,  3,  4,  6,  8, 10]]  # B




####################### FUNCTIONS #######################
def dump_tracks(midi_file):
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")

def dump_msgs(track):
    for i, msg in enumerate(track):
        print(str(i).rjust(4), ": ", msg, sep="")

def midi_to_music(midi_note):

    music_note = MUSIC_NOTES[midi_note % 12]
    octave = int(midi_note / 12) - 1

    return music_note, octave

def midi_to_string(midi_note):
    note = midi_to_music(midi_note)
    return note[0] + str(note[1])


_PROGRESS_BAR_LAST_I = 100
def progress_bar(done, total, resolution = 0):
    """
    Prints a progress bar to stdout.
    :param done: Number of items complete
    :param total: Total number if items
    :param resolution: How often to update the progress bar (in percentage).  0 will update each time
    :return: None
    """

    global _PROGRESS_BAR_LAST_I
    # percentage done
    i = int(done / total * 100)
    if i == _PROGRESS_BAR_LAST_I and resolution:
        return

    # if it's some multiple of resolution
    if (not resolution) or (not i % resolution) or (i == 100):
        stdout.write('\r')
        # print the progress bar
        stdout.write("[{}]{}%".format(("-" * int(i / 2) + (">" if i < 100 else "")).ljust(50), str(i).rjust(4)))
        # print the text figures
        stdout.write("({}/{})".format(done, total).rjust(13))
        stdout.flush()

    if i == 100:
        print("\nComplete!")

    _PROGRESS_BAR_LAST_I = i
