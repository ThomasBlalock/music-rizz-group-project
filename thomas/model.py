# instantiates a model class. example code below
#%%
import numpy as np

class Model:
    def __init__(self, tr_pr_filepath='../data/tr_pr.npy', states_filepath='../data/states.npy'):
        self.tr_pr = np.load(tr_pr_filepath)
        self.states = np.load(states_filepath).tolist()
    
    def __call__(self, start_state='start/end', temp=1.0, max_chords = 16):
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
    
#%%
model = Model()
model()