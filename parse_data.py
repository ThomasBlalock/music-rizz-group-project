#%%
import pandas as pd
import numpy as np

df = pd.read_csv('data/chordonomicon.csv')

def parse_chord_string(chord_string: str) -> dict:
    """
    Parses a string of tagged chord sections into a dictionary.

    Args:
        chord_string: A string with sections marked by <tags>
                      followed by space-separated chords.

    Returns:
        A dictionary where each key is the tag name (e.g., 'verse_1')
        and each value is a list of the chords in that section.
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