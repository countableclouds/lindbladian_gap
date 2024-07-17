import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import filter
import graph
import data
import lindbladian
import figure
from scipy.linalg import block_diag



beta = 1
gaps_ = []

GRAPH= 'cyclcic'
FILTER = 'ckg_metropolis'
SLOPE = 0
NAME = f"{FILTER}_{GRAPH}"

F = filter.Filter.from_name(FILTER)(1)
LOAD = False
SAVE = False

if LOAD:
    gaps_ = data.load(NAME)

#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 2
end = 20
pbar = tqdm(total=end-start)

if LOAD:
    print('Loaded.')
else:
    for n in range(start, end):
        start_time = time.time()
        
        M = graph.CyclicGraph.adj_matrix(n)
        N= np.array(graph.CyclicGraph.adj_matrix(n))/np.sqrt(2)
        # N= np.identity(n)
        # G = graph.Graph.from_name(GRAPH)(n, 'adjacent')
        G = graph.Graph.from_adjacency(M)(N)
        L = lindbladian.Lindbladian(G, F)
        L.initialize()
        L.cyclic_reshape()
        
        gaps_.append(L.spectral_gap())
        # print(gaps_)
        elapsed_time = time.time() - start_time
        pbar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f}s"})
        pbar.update(1)
        
        
    if SAVE:
        data.save(gaps_, NAME)
        print('Data Saved.')

pbar.close()

xs = np.array(range(start, end))
gaps = np.array(gaps_)
print(gaps)
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