# Composer Classifier
Mark Evers's Capstone Project



### Overview
A piano can play only 88 possible notes and within any song, only a fraction of those notes are actually used.  Yet somehow, composers are still able to express themselves with their own unique, recognizable style.  Familiar listeners are even capable of recognizing their favorite composers upon hearing a new song for the first time.  Composer Classifier uses a recursive neural network to learn composers' styles and predict the composer of a song it has never heard before.

### Data
There is a data protocol and associated file format called the Musical Instrument Digital Interface (MIDI) that electronic musical instruments use to communicate with one another.  There are many different types of commands these instruments can send to one another, but the most basic one tells an instrument to play a note.  The command contains a code for which note to play, such as `middle C`, alongside with a second instruction code `key down` (as in a piano key).  An electric piano would then begin to play its middle C note until it received a second MIDI command, `middle C -> key up`.  A sequence of MIDI messages can be saved to and replayed from a MIDI file (*.mid), which is where I will be getting my raw data from.

It is important to note that MIDI is strictly a data protocol and does not contain any recordings of actual audio, it is simply a list of instructions.  In essence, it is sheet music for computers.  Think of the difference between a recording of *Let It Be* by The Beatles and the sheet music for *Let It Be*.  My data would be the written notes contained in the sheet music, not the sonic rendering of the song as performed by The Beatles.

### Feature Extraction
The raw sequence of MIDI messages must be converted into a more usable format.  Every single note in the MIDI file is really two instructions, a `key down` and an associated `key up`.  These must be converted into a tuple consisting of `(key, duration)` and saved in a list.  When no notes are being played, a musical rest of appropriate length must be appended to the list.  This can be accomplished with aid of the `mido` python library.  It can be installed with pip:
```bash
pip install mido
```
