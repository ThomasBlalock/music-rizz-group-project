#%%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
# Configuration section
#
datadir = 'data'
# support developer specific directories in our repo
if not Path(datadir).is_dir() and Path(f'../{datadir}').is_dir():
    datadir = f'../{datadir}'
    print(f'{__name__} using data dir {Path(datadir).resolve()}')
else:
    print(f'{__name__} using data dir {Path(datadir).resolve()}')

files = {
    'chordonomicon_parsed' : f'{datadir}/chordonomicon_parsed.csv',
    'chordonomicon_parsed_pickled' : f'{datadir}/chordonomicon_parsed.pkl',
}

# if Path(files['chordonomicon_parsed_pickled']).is_file():
#     df = pd.read_pickle(files['chordonomicon_parsed_pickled'])
#     print(f'{__name__} loaded {files['chordonomicon_parsed_pickled']}')

# else:
df = pd.read_csv(f'{datadir}/chordonomicon.csv', low_memory=False)

def parse_chord_string(chord_string: str) -> dict:
    """
    Args:
        chords (str): The 'chords' column from the chordonomicon

    Returns:
        {'<tag>': ['chord1', 'chord2', ...], ...}
    """
    chord_map = {}
    current_key = None
    current_chords = []
    parts = chord_string.split()
    for part in parts:
        if part.startswith('<') and part.endswith('>'):
            if current_key is not None:
                chord_map[current_key] = current_chords

            current_key = part[1:-1]
            current_chords = []
        else:
            if current_key is None:
                current_key = 'unknown'
            current_chords.append(part)

    if current_key is not None and current_chords:
        chord_map[current_key] = current_chords

    return chord_map

df['chord_map'] = df['chords'].apply(parse_chord_string)
df.to_csv(f'{datadir}/chordonomicon_parsed.csv', index=False)
#df.to_pickle(f'{datadir}/chordonomicon_parsed.pkl')