import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import filter
import graph
import data
import lindbladian
import figure


beta = 1
gaps_ = []

GRAPH= 'pathh'
FILTER = 'ckg_metropolis'
SLOPE = 0
NAME = f"{FILTER}_{GRAPH}"

F = filter.Filter.from_name(FILTER)(1)
LOAD = False
SAVE = False

if LOAD:
    gaps_ = data.load(NAME)

#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 3
end = 50

if LOAD:
    print('Loaded.')
else:
    for n in tqdm(range(start, end)):
        M = graph.PathGraph.adj_matrix(n)
        G = graph.Graph.from_name(GRAPH)(M, n, "diagonal")
        L = lindbladian.Lindbladian(G, F)
        gaps_.append(L.spectral_gap(assertion=True))
        
    if SAVE:
        data.save(gaps_, NAME)
        print('Data Saved.')

print(gaps_)

xs = np.array(range(start, end))
gaps = np.array(gaps_)
ln_xs = np.log(xs)
ln_gaps = np.log(gaps)
b = np.mean(ln_gaps[-3:] - SLOPE*ln_xs[-3:])

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (n)', 'Spectral Gap']
figure.plot(xs, gaps, ax, labels)
if SAVE:
    fig.savefig(f"figures/{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (ln n)', 'Log Spectral Gap']
if SLOPE:
    figure.plot(ln_xs, ln_gaps, ax, labels, line=[SLOPE, b])
else:
    figure.plot(ln_xs, ln_gaps, ax, labels)
if SAVE:
    fig.savefig(f"figures/lnln_{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

plt.show()