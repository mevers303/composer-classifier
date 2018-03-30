# Mark Evers
# Created: 3/30/2018
# globals.py
# Global variables and functions


MINIMUM_WORKS = 100
MAXIMUM_WORKS = 100
NUM_THREADS = 2



import sys

PROGRESS_BAR_LAST_I = 100
def progress_bar(done, total, resolution = 1):

    global PROGRESS_BAR_LAST_I
    # percentage done
    i = int(done / total * 100)
    if i == PROGRESS_BAR_LAST_I:
        return

    # if it's some multiple of resolution
    if (not i % resolution) or (i == 100):
        sys.stdout.write('\r')
        sys.stdout.write("[{}] {}% ({}/{})".format(('=' * int(i / 2) + '>').ljust(50), str(i).rjust(4), done, total))
        sys.stdout.flush()

    if i == 100:
        print("\nComplete!")

    PROGRESS_BAR_LAST_I = i
