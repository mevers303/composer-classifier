# Mark Evers
# Created: 4/2/2018
# midi_archive.py
# Functions for processing MIDI files


import mido
import os
import pandas as pd
import threading
import numpy as np

from globals import *




class MidiArchive():
    """
    Class for building the metadata for a MIDI archive.
    """

    def __init__(self, base_dir="midi/"):
        """
        :param base_dir: The base directory where the MIDI files are contained in composer subfolders.
        """

        self.base_dir = base_dir
        self.composers = set()

        self.midi_filenames = []
        self.midi_filenames_labels = []
        self.midi_filenames_total = 0
        self.midi_filenames_invalid = []
        self.midi_filenames_parsed = 0

        columns = ["composer", "type", "tracks", "ticks_per_beat", "first_key_sig", "first_time_n", "first_time_d", "first_time_32nd", "time_clocks_per_click", "first_note", "first_note_time", "has_note_off", "has_key_change"]
        columns.extend(MUSIC_NOTES)
        # columns.extend(["midi_" + str(i) for i in range(128)])
        self.meta_df = pd.DataFrame(columns=columns)
        self.meta_df.index.name = "filename"

        self.threads = []
        self.thread_lock = None
        self.stop_threads = False

        # self.key_sigs = set()
        # self.time_sigs = set()



    def get_all_filenames(self):
        """
        Returns a list of files in <self.base_dir> and their associated label in y.  Files must be in
        <dir>/<composer>/*.mid"
        :return: None
        """""

        self.midi_filenames = []
        self.midi_filenames_labels = []
        self.composers = set()


        for composer in os.listdir(self.base_dir):


            composer_files = []
            for root, dirs, files in os.walk(os.path.join(self.base_dir, composer)):

                for file in files:

                    full_path = os.path.join(root, file)
                    if not (file.lower().endswith(".mid") or file.lower().endswith(".midi")):
                        print("Unknown file:", full_path)
                        continue

                    composer_files.append(full_path)


            composer_works = len(composer_files)
            # if composer_works < MINIMUM_WORKS:
            #     # print("Not enough works for {}, ({})".format(composer, composer_works))
            #     continue
            # if composer_works > MAXIMUM_WORKS:
            #     composer_files = random.sample(composer_files, MAXIMUM_WORKS)
            #     composer_works = len(composer_files)


            self.midi_filenames.extend(composer_files)
            self.midi_filenames_labels.extend([composer] * composer_works)
            self.composers.add(composer)
            # print("Added {} ({})".format(composer, composer_works))


        self.midi_filenames_total = len(self.midi_filenames)
        print("Found", self.midi_filenames_total, "files from", len(self.composers), "composers!")

        return self.midi_filenames, self.midi_filenames_labels



    def build_meta_df(self):
        """
        Builds the meta data pandas dataframe
        :return: None
        """

        chunk_size = int(self.midi_filenames_total / NUM_THREADS + 1)
        chunkified_filenames = [self.midi_filenames[i:i + chunk_size] for i in
                                range(0, len(self.midi_filenames), chunk_size)]
        chunkified_labels = [self.midi_filenames_labels[i:i + chunk_size] for i in
                             range(0, len(self.midi_filenames_labels), chunk_size)]


        print("Loading midi files with", len(chunkified_filenames), "threads...")
        self.thread_lock = threading.Lock()
        self.stop_threads = False

        for filenames, labels in zip(chunkified_filenames, chunkified_labels):
            thread = threading.Thread(target=MidiArchive.build_meta_df_chunk,
                                      args=(self, filenames, labels))
            thread.start()
            self.threads.append(thread)

        try:
            for thread in self.threads:
                thread.join()

        except KeyboardInterrupt:
            self.stop_threads = True
            raise KeyboardInterrupt

        self.threads = []
        self.thread_lock = None


        return self.meta_df



    def build_meta_df_chunk(self, filenames, labels):
        """
        :param filenames: list of paths to MIDI files
        :param labels: list of labels (composers) for the MIDI files
        :return: None

        Gets the metadata for a list of files.  This exists as a chunk to work with threading.
        """

        for file, composer in zip(filenames, labels):
            if self.stop_threads:
                break
            self.parse_midi_meta(file, composer)

        # with self.thread_lock:
        #     print("\nThread finished!")



    def parse_midi_meta(self, file, composer):
        """
        Adds a MIDI file's metadata to the meta_df pandas dataframe.
        :param file: path to a MIDI file
        :param composer: the label (composer) for this file
        :return: None
        """

        key_sig = time_n = time_d = time_32nd = time_clocks_per_click = first_note = first_note_time = None
        has_note_off = has_key_change = False
        music_notes_before_key_change = np.zeros((12,))
        # midi_notes_before_key_change = np.zeros((128,))

        time_now = 0
        last_key_change_time = 0
        last_key = None

        try:
            mid = mido.MidiFile(file)

            for msg in mid:
                time_now += msg.time


                if msg.type == "key_signature":
                    if msg.key == last_key:
                        continue

                    # self.key_sigs.add(msg.key)
                    # if 0 < time_now - last_key_change_time < 5:
                    #     print("\nVery short key signature change ({}s) -> {}".format(time_now - last_key_change_time, mid.filename))

                    if not key_sig or not music_notes_before_key_change.sum():
                        key_sig = msg.key
                    elif time_now - last_key_change_time != 0:  # if the time since the last key change is zero
                        has_key_change = True

                    last_key_change_time = time_now
                    last_key = msg.key


                elif msg.type == "time_signature":
                    # self.time_sigs.add("{}/{}/{}/{}".format(msg.numerator, msg.denominator, msg.notated_32nd_notes_per_beat, msg.clocks_per_click))
                    if not time_n:
                        time_n = msg.numerator
                        time_d = msg.denominator
                        time_32nd = msg.notated_32nd_notes_per_beat
                        time_clocks_per_click = msg.clocks_per_click


                elif msg.type == "note_on":
                    if not msg.velocity:
                        continue
                    if not first_note:
                        first_note = midi_to_music(msg.note)[0]
                        first_note_time = msg.time
                    if not has_key_change and msg.channel != 10:  # skip channel 10 (drums)
                        # midi_notes_before_key_change[msg.note] += 1
                        music_notes_before_key_change[msg.note % 12] += 1


                elif msg.type == "note_off":
                    if not has_note_off:
                        has_note_off = True


            with self.thread_lock:

                values = [composer, mid.type, len(mid.tracks), mid.ticks_per_beat, key_sig, time_n, time_d, time_32nd, time_clocks_per_click, first_note, first_note_time, has_note_off, has_key_change]
                values.extend(music_notes_before_key_change)
                # values.extend(midi_notes_before_key_change)
                self.meta_df.loc[file] = values

                self.midi_filenames_parsed += 1
                progress_bar(self.midi_filenames_parsed, self.midi_filenames_total)


        except KeyboardInterrupt:
            # skip the next except clause
            raise KeyboardInterrupt

        except:
            with self.thread_lock:
                print("\nERROR -> Skipping invalid file:", file)
                self.midi_filenames_invalid.append(file)
                self.midi_filenames_parsed += 1
                progress_bar(self.midi_filenames_parsed, self.midi_filenames_total)




def build_all_meta(dir="midi", delete_invalid_files=False):
    """
    Creates a csv file containing the metadata for a directory containing MIDI files organized into folders named after
    their composer
    
    :param dir: The path to the base directory of the archive.
    :param delete_invalid_files: Whether or not to delete invalid MIDI files from the system.
    :return: None
    """

    archive = MidiArchive(dir)
    archive.get_all_filenames()
    df = archive.build_meta_df()

    if delete_invalid_files:
        for file in archive.midi_filenames_invalid:
            archive.midi_filenames.remove(file)
            os.remove(file)
            print("Deleted corrupt file <", file, ">")


    print("Saving meta csv...")
    df.to_csv(os.path.join(dir, "meta.csv"))
    print("Meta CSV file saved!")

    # info = {"key_sigs": list(archive.key_sigs), "time_sigs": list(archive.time_sigs)}
    # with open(os.path.join(dir, "info.json"), "w") as f:
    #     json.dump(info, f)
    # print("JSON file saved!")


def get_meta_df(dir="midi"):
    """
    Gets a meta dataframe with the proper schema from <dir>.
    :param dir: The base directory of the archive.
    :return: A pandas dataframe with the metadate.
    """
    return pd.read_csv(os.path.join(dir, "meta.csv"), index_col="filename")





if __name__ == "__main__":
    
    from sys import argv
    delete_invalid_files = False
    
    if len(argv) == 1:
        build_all_meta()
        
    else:
    
        for arg in argv[1:]:
            
            if arg == "--delete-corrupt-files":
                delete_invalid_files = True
                continue
                
            if os.path.isdir(arg):
                build_all_meta(arg, delete_invalid_files)
            else:
                print(arg, "is not a valid directory!")
                print("Usage:\n  python midi_archive.py [--delete-corrupt-files] <archive_dir1> <archive_dir2> ...")
