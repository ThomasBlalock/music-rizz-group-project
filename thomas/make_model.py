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
tr_counts = np.zeros( (S+2, S+2) )

for t in range(1,T): # For each transition
    # Current and next tokens:
    x_tm1 = seq[t-1] # previous state
    x_t = seq[t] # current state
    # Determine transition indices:
    index_from = states.index(x_tm1)
    index_to = states.index(x_t)
    # Update transition counts:
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
order = 3
sq = [''.join(seq[(t-order-1):(t-1)]) for t in range(order+1, T)]

states = list(np.unique(sq))
print('States: \n', states)
S = len(states)
T = len(sq)

tr_counts = np.zeros( (S, S) )
for t in range(1,T):
    x_previous = sq[t-1] # previous state
    x_next = sq[t] # current token

    index_from = states.index(x_previous)
    index_to = states.index(x_next)

    tr_counts[index_to, index_from] += 1

print(f'Transition Counts:\n {tr_counts}')

#%%
sums = tr_counts.sum(axis=1)
print('State proportions: \n')

tr_df = pd.DataFrame(sums/np.sum(sums,axis=0), index=states)
print(tr_df)

#%%
tr_pr = np.divide(tr_counts, sums, 
                             out=np.zeros_like(tr_counts), 
                             where=sums!=0)

print('Transition Proportions:\n')

tr_df = pd.DataFrame(np.round(tr_pr,2), index=states, columns=states)
print(tr_df)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(tr_pr, 
            cmap='Blues',       # Or 'Blues', 'plasma', whatever looks good
            square=True,          # Keep cells square
            xticklabels=states,
            yticklabels=states,
            cbar_kws={'label': 'Transition Probability'})

plt.title('Transition Probabilities')
plt.xlabel('...To State')
plt.ylabel('From State...')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
