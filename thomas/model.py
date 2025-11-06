# instantiates a model class
#%%
import numpy as np
from chord_map import notes_map
from IPython.display import Audio, display
import time

class Model:
    def __init__(self, tr_pr_filepath='../data/tr_pr.npy', states_filepath='../data/states.npy'):
        self.tr_pr = np.load(tr_pr_filepath)
        self.states = np.load(states_filepath).tolist()
    
    def generate_chords(self, start_state='start/end', temp=1.0, max_chords = 16):
        state_idx = self.states.index(start_state)
        while True:
            probs = self.tr_pr[state_idx]
            probs = np.power(probs, 1.0 / temp)
            probs = probs / np.sum(probs)
            next_state = np.random.choice(self.states, size=1, p=probs)[0]
            if next_state == 'start/end' or max_chords == 0:
                break
            max_chords -= 1
            yield next_state
            state_idx = self.states.index(next_state)

    def _generate_chord_audio(self, midi_notes, duration, volume=0.3, sample_rate=22050):
        """Generates the numpy array for a single chord."""
        if not midi_notes: return np.zeros(int(sample_rate * duration))
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.zeros(len(t))
        for note_num in midi_notes:
            freq = 440.0 * (2.0 ** ((note_num - 69.0) / 12.0))
            tone += np.sin(2 * np.pi * freq * t)
            
        tone = (tone / len(midi_notes)) * volume
        
        # Quick fade in/out to avoid clicks between chords
        fade_len = int(0.02 * sample_rate)
        if len(t) > fade_len * 2:
            tone[:fade_len] *= np.linspace(0, 1, fade_len)
            tone[-fade_len:] *= np.linspace(1, 0, fade_len)
            
        return tone

    def __call__(self, bpm=120, start_state='start/end', temp=1.0, max_chords = 16):
        spb = 60.0 / bpm
        time_left = spb * 4
        full_audio = []
        chords_list = []
        for chords in self.generate_chords(start_state=start_state, 
                                               temp=temp, 
                                               max_chords=max_chords):
            midi_notes = notes_map(chords)
            chords_list.append(chords)
            
            if midi_notes:
                dur_candidates = [can for can in [spb, spb*2.0, spb*4.0] if can <= time_left]
                duration = np.random.choice(dur_candidates)
                time_left -= duration
                if time_left <= 0:
                    time_left = spb * 4
                chord_audio = self._generate_chord_audio(midi_notes, duration)
                full_audio.append(chord_audio)
        
        if full_audio:
            combined_track = np.concatenate(full_audio)
            display(Audio(data=combined_track, rate=22050, autoplay=True))
        return chords_list