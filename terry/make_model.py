# Makes the model and makes some visualizations
# %%
# Imports
#
import os
import sys
from pathlib import Path
# Add parent directory to search path in case we're in a user folder
# but do it only once, in case we're in an interactive environment.
if '..' not in sys.path:
    #print(f'Search path: {"\n".join(sys.path)}')
    #sys.path.insert(1+sys.path.index(""), '..') # This works only for interactive, not from cmd line
    sys.path.append('..')
# print(f'Current working directory: {os.getcwd()}') 
# print(f'Search path: {"\n".join(sys.path)}')
import pandas as pd
import numpy as np
import pickle
from thomas.chord_map import uncommon_chord_map

def make_model(genres= ['pop', 'rock', 'metal', 'country', 'punk', 'alternative', 'jazz']):
    # %%
    # Configuration section
    #
    datadir = 'data'
    # support developer specific directories in our repo
    if not Path(datadir).is_dir() and Path(f'../{datadir}').is_dir():
        datadir = f'../{datadir}'
        print(f'Using data dir {Path(datadir).resolve()}')
    else:
        print(f'Using data dir {Path(datadir).resolve()}')

    files = {
        'chordonomicon': f'{datadir}/chordonomicon.csv', 
        'chordonomicon_pickled': f'{datadir}/chordonomicon_parsed.pkl',
        'statrans_pickled': f'{datadir}/states_transitions_GENRE.pkl',
        'tr_pr': f'{datadir}/tr_pr_GENRE.npy',
        'states': f'{datadir}/states_GENRE.npy',
    }


    # %% 
    # Load the chordonomicon
    #
    if Path(files['chordonomicon_pickled']).is_file():
        print(f'Loading {files["chordonomicon_pickled"]}')
        cdf = pd.read_pickle(files['chordonomicon_pickled'])
    else:
        from parse_data import parse_chord_string
        cdf = pd.read_csv(files['chordonomicon'], low_memory=False)
        cdf['chord_map'] = cdf['chords'].apply(parse_chord_string)
        cdf.to_pickle(files['chordonomicon_pickled'])

    for genre in genres:
        # %%
        # Identify states and transitions
        #
        states_path = files['statrans_pickled'].replace('GENRE', genre)
        if Path(states_path).is_file():
            with open(states_path, "rb") as file: 
                states, seq, S, T = pickle.load(file)
            print(f'Loaded {states_path}')
        else:
            df = cdf[cdf['main_genre'] == genre]
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
            with open(states_path, "wb") as file:
                pickle.dump((states, seq, S, T), file)
                print(f'Saved {states_path}')

        #%%
        # Create a S X S transition matrix, and find the transition counts:
        #
        print(f'Generating Transition Matrix for {genre} genre')
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
        # Plot the transition probabilities
        #
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(13, 11))
        sns.heatmap(tr_pr, 
                    cmap='Blues',
                    square=True,          
                    xticklabels=states,
                    yticklabels=states,
                    cbar_kws={'label': 'Transition Probability'})

        plt.title(f'Transition Probabilities for {genre}')
        plt.xlabel('...To State')
        plt.ylabel('From State...')
        #plt.show()

        #%%
        # Save the model
        #
        np.save(files['tr_pr'].replace('GENRE', genre) , tr_pr)
        np.save(files['states'].replace('GENRE', genre), states)

    # end genre loop