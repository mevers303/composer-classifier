# Composer Classifier
Mark Evers



### Overview
A piano can play only 88 possible notes and within any song, only a fraction of those notes are actually used.  Yet somehow, composers are still able to express themselves with their own unique, recognizable style.  Familiar listeners are even capable of recognizing their favorite composers upon hearing a new song for the first time.  Composer Classifier mimics this ability by useing a recursive neural network to learn composers' styles and predict the composer of a song it has never heard before.


### Data Sources
There is a data protocol and associated file format called the Musical Instrument Digital Interface (MIDI) that electronic musical instruments use to communicate with one another.  When sheet music is composed or transcribed by a musician on computer, the software (such as Sibelieus) are capable of exporting a MIDI file.  A quick google search for "MIDI file archive" returns numerous results.  Much of these archives are compiled from contributions from users who have done exactly that.

Here are the sources of the MIDI files:
[Reddit Post](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/)

[ClassicalMidi.co.uk](https://www.classicalmidi.co.uk/page7.htm)

[Download-Midi.com](http://www.download-midi.com/)


### About MIDI
There are many different types of commands these instruments can send to one another, but the most basic one tells an instrument to play a note.  This command contains a code for which note to play, such as `middle C`, alongside a second code for `key down` (as in a piano key).  An electric piano would then begin to play its middle C note until it received a second MIDI command, `middle C -> key up`.  A sequence of MIDI messages can be saved to and replayed from a MIDI file (\*.mid), which is where we will be getting our raw data from.

It is important to note that MIDI is strictly a data protocol and does not contain any recordings of actual audio, it is simply a list of instructions.  In essence, it is sheet music for computers.  Think of the difference between a recording of *Let It Be* by The Beatles and the sheet music for *Let It Be*.  Our data would be the written notes contained in the sheet music, not the sonic rendering of the song as performed by The Beatles.


### Feature Extraction
The raw sequence of MIDI messages must be converted into a more usable format.  Any single note in the MIDI file is really two instructions, a `key down` and an associated `key up`.  These must be converted into a tuple consisting of `(key, duration)` and saved into a list.  When no notes are being played, a musical rest of appropriate length must be appended to the list.  This can be accomplished with aid of the `mido` python library.  It can be installed with pip:
```bash
pip install mido
```
The `mido` library is pivitol for this application.  It provides objects for reading and handling MIDI files, tracks, and messages.  Visit the [Mido documentation](https://mido.readthedocs.io/en/latest/midi_files.html) for more info.

There are 3 feature extraction strategies we will be exploring.  Now we will demonstrate how the following example MIDI messages will be changed with each strategy:
```
<note_on  channel=0 note=45 velocity=110 time=0>
<note_off channel=0 note=45 velocity=0   time=256>
<note_on  channel=0 note=47 velocity=110 time=0>
<note_off channel=0 note=47 velocity=0   time=256>
```

##### 1.  Text Encoding
Each MIDI track is converted into a text string with the format `"<note_name>:<duration>"`.  The example MIDI messages will be encoded as such:
```
"TRACK_START A1:256 B1:256 TRACK_END"
```

##### 2.  n-Hot Encoding
Each MIDI track will be converted to a time-series encoded as a n-hot vector:

| Time | C | C# | D | D# | E | F | F# | G | G# | A | A# | B |
| ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |


##### 3.  Raw MIDI Messages
Each MIDI track will be converted into a vector containing the raw message data.  I'm not sure this will work, but let's try it why not:

| time | channel | note | velocity | time |
| --- | --- | --- | --- | --- |
| 1 | 0 | 45 | 110 | 0 |
| 2 | 0 | 47 | 110 | 256 |


### Local Dataset
The MIDI files to be used must be stored with a specific directory structure.  They must be stored in a single folder.  The files must be arranged in sub-folders by their composer/label.  Any subfolders beyond that will not affect the labeling or functions, all their contents will be scanned.  Any files with the `.mid` or `.midi` extension.
```
<root dir>
  -> Bach
  ----> Cello Suite No 2.mid
  ----> Cello Suite No 3.mid
  -> Beethoven
  ----> Piano Sonata No 2.mid
  ----> Piano Sonata No 27.mid
  -> Chopin
  ----> Waltz Op 64 No 1.mid
  ----> Waltz Op 64 No 2.mid
```

There are several functions in `file_functions.py` that may assist you.

##### Meta Dataframe
A Pandas dataframe containing metadata about the local MIDI files must be created and saved as a .csv for later use.  The `midi_archive.py` script will create it for you.  Pass the base directory containg the composers subdirectories as an argument.
```bash
python src/midi_archive.py midi/classical/
```

### Model
Composer Classifier uses a long short term memory recursive neural network (LSTM RNN).
