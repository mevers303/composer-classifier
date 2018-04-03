import pandas as pd
from src.midi_archive import get_meta_df

all_df = get_meta_df("raw_midi")

composers = pd.DataFrame(all_df.groupby("composer").type.count())
composers.columns = ["works"]

