# Mark Evers
# Created: 3/30/2018
# globals.py
# Global variables and functions

from sys import stdout
from numpy import argsort



####################### OPTIONS #########################
######### FEATURES
# How many pieces must a composer have for us to consider them?
MINIMUM_WORKS = 100
# How many pieces will we use from each composer?
MAXIMUM_WORKS = 150

###### HYPER PARAMETERS
# How many threads to use when parsing the MIDI archive?
NUM_THREADS = 3
# How many ticks per beat should each track be converted to?
TICKS_PER_BEAT = 1024
# The resolution of music notes
NOTE_RESOLUTION = 32
# The longest note allowed
MAXIMUM_NOTE_LENGTH = TICKS_PER_BEAT * 8
# Look at the first x notes to train/classify
NUM_STEPS = 50
# The number of unique features to use in the CountVectorizer.
TEXT_MAXIMUM_FEATURES = 30000
# How many midi files to load at once
BATCH_SIZE = 78
# How many epochs to train for?
N_EPOCHS = 50
# The nodes in each hidden layer
HIDDEN_LAYER_SIZE = 512



####################### CONSTANTS #######################
#               0     1    2    3     4    5    6     7    8     9    10    11
MUSIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_SIGNATURES = [[ 0,  2,  4,  5 , 7,  9, 11],  # C
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
    """
    Prints all the tracks in a mido MidiFile object.

    :param midi_file: A mido.MidiFile object.
    :return: None
    """
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")

def dump_msgs(mido_object):
    """
    Prints all the messages contained in a track or midi file.  The delta time is in seconds when you give it a
    mido.MidiFile.

    :param track: A mido MidiTrack or MidiFile
    :return: None
    """
    for i, msg in enumerate(mido_object):
        print(str(i).rjust(4), ": ", msg, sep="")

def midi_to_music(midi_note):
    """
    Returns a tuple of (<note name>, octave).

    :param midi_note: The midi note value.
    :return: (<string> Note Name, <int> Octave)
    """

    music_note = MUSIC_NOTES[midi_note % 12]
    octave = int(midi_note / 12) - 1

    return music_note, octave

def midi_to_string(midi_note):
    """
    Converts a MIDI note to a string of it's name and octave (like C4 for middle C).

    :param midi_note: The midi note value
    :return: A string as described above.
    """
    note = midi_to_music(midi_note)
    return note[0] + str(note[1])

def get_key_sig(note_dist):
    """
    Uses the note distribution in the meta dataframe to determine a filename's key signature
    :param note_dist: Distribution of notes as per music_notes
    :return: The index of key_signatures that is the best match.
    """

    top_notes = set(argsort(note_dist)[::-1][:7])


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


TICK_RESOLUTION = TICKS_PER_BEAT / NOTE_RESOLUTION
def bin_note_duration(duration):

    binned = int(duration / TICK_RESOLUTION)
    remainder = duration % TICK_RESOLUTION

    if remainder > TICK_RESOLUTION / 2:
        binned += 1

    return int(binned * TICK_RESOLUTION)




_PROGRESS_BAR_LAST_I = 100
def progress_bar(done, total, resolution = 0, text = ""):
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
        stdout.write("({}/{})".format(done, total).rjust(15))
        if text:
            stdout.write(" " + text)
        stdout.flush()

    if i == 100:
        print("\nComplete!")

    _PROGRESS_BAR_LAST_I = i
