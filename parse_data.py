#%%
import pandas as pd
import numpy as np

df = pd.read_csv('data/chordonomicon.csv')

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
df.to_csv('data/chordonomicon_parsed.csv', index=False)