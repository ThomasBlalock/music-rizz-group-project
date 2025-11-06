# Makes the model and makes some visualizations
#%%
import os
os.chdir('..')
import pandas as pd
import numpy as np
from parse_data import parse_chord_string
from thomas.chord_map import uncommon_chord_map

df = pd.read_csv('data/chordonomicon.csv')
df['chord_map'] = df['chords'].apply(parse_chord_string)

#%%
df = df[df['main_genre'] == 'pop']
seq = []
states = set()
for chord_map in df['chord_map'].tolist():
    for k, v in chord_map.items():
        if 'chorus' not in k:
            continue
        new_chords = ['start/end']
        for c in v:
            if '/' in c: # The / is what the bass does, semantically safe to remove
                # uncommon_chords_map maps the least-frequent chords to similar chords
                new_chords.append(uncommon_chord_map(c.split('/')[0]))
            else:
                new_chords.append(uncommon_chord_map(c))
        states.update(new_chords)
        seq.extend(new_chords)
states = list(states)
seq = seq + ['start/end']
S = len(states)
T = len(seq)
print(S, T)

#%%
## Create a S X S transition matrix, and find the transition counts:
tr_counts = np.zeros( (S, S) )

for t in range(1,T):
    x_prev, x_curr = seq[t-1], seq[t]
    index_from = states.index(x_prev)
    index_to = states.index(x_curr)
    tr_counts[index_to, index_from] += 1

print(f'\nTransition Counts:\n {tr_counts}')

# Sum the transition counts by row:
sums = tr_counts.sum(axis=1, keepdims=True)
print(f'\nState Counts: \n {sums}')

# Sum the transition counts by row:
print(f'\nState proportions: \n {sums/np.sum(sums)}')

# Normalize the transition count matrix to get proportions:
tr_pr = np.divide(tr_counts, sums, 
                            out=np.zeros_like(tr_counts), 
                            where= sums!=0)

tr_pr_1 = tr_pr # Save transition matrix for later

print(f'\nTransition Proportions:')
pd.DataFrame(np.round(tr_pr,2), index=states, columns=states)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 11))
sns.heatmap(tr_pr, 
            cmap='Blues',
            square=True,          
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.show()

#%%
np.save('data/tr_pr.npy', tr_pr)
np.save('data/states.npy', states)