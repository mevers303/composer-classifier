# Mark Evers
# Created: 3/30/2018
# globals.py
# Global variables and functions


# How many pieces must a composer have for us to consider them?
MINIMUM_WORKS = 100
# How many pieces will we use from each composer?
MAXIMUM_WORKS = 200

# How many threads to use when parsing the MIDI archive?
NUM_THREADS = 2
# How many ticks per beat should each track be converted to?
TICKS_PER_BEAT = 1024



import sys

PROGRESS_BAR_LAST_I = 100
def progress_bar(done, total, resolution = 0):
    """
    :param done: Number of items complete
    :param total: Total number if items
    :param resolution: How often to update the progress bar (in percentage).  0 will update each time
    :return: None

    Prints a progress bar to stdout.
    """

    global PROGRESS_BAR_LAST_I
    # percentage done
    i = int(done / total * 100)
    if i == PROGRESS_BAR_LAST_I and resolution:
        return

    # if it's some multiple of resolution
    if (not resolution) or (not i % resolution) or (i == 100):
        sys.stdout.write('\r')
        # print the progress bar
        sys.stdout.write("[{}]{}%".format(("-" * int(i / 2) + (">" if i < 100 else "")).ljust(50), str(i).rjust(4)))
        # print the text figures
        sys.stdout.write("({}/{})".format(done, total).rjust(13))
        sys.stdout.flush()

    if i == 100:
        print("\nComplete!")

    PROGRESS_BAR_LAST_I = i
