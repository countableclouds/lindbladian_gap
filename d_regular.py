import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import filter
import graph
import data
import lindbladian
import figure

def regular_sample(d, n, seed):
    while True:
        G = nx.random_regular_graph(d, n, seed)
        if nx.is_connected(G):
            break
    A = nx.adjacency_matrix(G).toarray()
    return G, np.matrix(A)
    
beta = 1
gaps_ = []

GRAPH= '4_regular'
FILTER = 'ckg_metropolis'
SLOPE = -1
NAME = f"{FILTER}_{GRAPH}"

F = filter.Filter.from_name(FILTER)(1)


LOAD = True
SAVE = False
#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 5
end = 25
TRIALS = 100
d = 4
seed = 0
err= []

if LOAD:
    gaps_ = data.load(NAME)
    print('Loaded.')
else:
    for n in range(start, end, 5):
        trial_data = []
        for trial in range(TRIALS):
            if trial%10==0:
                print(n, trial)
            G_draw, M = regular_sample(d, n, hash((d, n, trial, seed)))
            G = graph.Graph.from_name(GRAPH)(M, n, "diagonal")
            L = lindbladian.Lindbladian(G, F)
            trial_data.append(L.spectral_gap(assertion=True))

        trial_data = [elem for elem in trial_data if round(elem, 10)!=0]
        
        gap = np.mean(trial_data)
        std = np.std(trial_data)
        print(gap, std, std / np.sqrt(TRIALS))
        err.append(std / np.sqrt(TRIALS))
        gaps_.append(gap)
        
    if SAVE:
        data.save(gaps_, NAME)
        print('Data Saved.')


gaps_ = [0.06234058259659126, 0.051196775407997046, 0.03832856850102053, 0.032226344375980626]


xs = np.array(range(start, end, 5))
gaps = np.array(gaps_)
ln_xs = np.log(xs)
ln_gaps = np.log(gaps)
b = np.mean(ln_gaps[-3:] - SLOPE*ln_xs[-3:])

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (n)', 'Spectral Gap']
figure.plot(xs, gaps, ax, labels)

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (ln n)', 'Log Spectral Gap']

figure.plot(ln_xs, ln_gaps, ax, labels, line=[SLOPE, b])

plt.show()