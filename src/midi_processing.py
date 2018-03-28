# Mark Evers
# Created: 3/27/2018
# midi_processing.py
# Functions for processing MIDI files

import mido



def dump_tracks(midi_file):
    for i, track in enumerate(midi_file.tracks):
        print(str(i).rjust(2), ": ", track, sep="")



def dump_msgs(track):
    for i, msg in enumerate(track):
        print(str(i).rjust(4), ": ", msg, sep="")



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

    mid = mido.MidiFile("midi/Bach/piano/Piano version of Bachs two part inventions No.4.mid")
