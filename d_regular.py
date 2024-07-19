import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
from tqdm import tqdm
import time
import filter
import graph
import data
import lindbladian
import figure




def regular_sample(d, n):
    counter = 0
    while True:
        G = nx.random_regular_graph(d, n)
        if nx.is_connected(G):
            break
        counter+=1
    A = nx.adjacency_matrix(G).toarray()
    return G, np.matrix(A)


def regular_sample_2(d, n):
    sampling = True

    assert n%2==0, "This sampling technique only works with an even number of vertices"
    
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    while True:
        ords = []
        for i in range(d):
            ords.append(list(range(0, n)))
            random.shuffle(ords[-1])
        edges = [tuple(ords[j//n][j%n:j%n+2]) for j in range(0, n*d, 2)]
        edge_colors = [[edges[(n//2)*j+i] for i in range(n//2)] for j in range(d)]
        if len(set(edges)) == len(edges):
            G.add_edges_from(edges)
            if nx.is_connected(G):
                coloring = dict(zip(list(range(0, d)), edge_colors))
                break
            else:
                G.clear_edges()
    A = nx.adjacency_matrix(G).toarray()
            
    return G, A, coloring
            
    

    
beta = 1
gaps_ = []

GRAPH= '2_regular'
FILTER = 'davies_metropolis'
SLOPE = 0
NAME = f"{FILTER}_{GRAPH}"

F = filter.Filter.from_name(FILTER)(1)


LOAD = False
SAVE = True
#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 4
end = 25
TRIALS = 100
d = 2
seed = 0
err= []

if LOAD:
    gaps_ = data.load(NAME)
    print('Loaded.')
else:
    for n in range(start, end, 2):
        # N= np.identity(n)
        try:
            trial_data = []
            for trial in tqdm(range(TRIALS)):
                G_draw, M, color = regular_sample_2(d, n)
                
                jumps = []
                for key in color.keys():
                    N= np.zeros((n, n))
                    for edge in color[key]:
                        N[edge[0], edge[1]] = 1
                    jumps.append(lindbladian.sparsify_jump(N))
                    jumps.append(lindbladian.sparsify_jump(N.conjugate().transpose()))

                # jumps = []
                # for i in range(n):
                #     N= np.zeros((n, n))
                #     N[i, i] = 1
                #     jumps.append(lindbladian.sparsify_jump(N))

                G = graph.Graph.from_adjacency(M)(jumps)
                
                L = lindbladian.Lindbladian(G, F)
                L.initialize()
                trial_data.append(L.spectral_gap(assertion=True))
                # if trial%25==0:
                #     print(n, trial, trial_data[-1])
                
            trial_data = [elem for elem in trial_data if round(elem, 10)!=0]
            
            gap = np.mean(trial_data)
            std = np.std(trial_data)
            print(n, gap, std, std / np.sqrt(TRIALS),std / np.sqrt(TRIALS)/gap*100)
            err.append(std / np.sqrt(TRIALS))
            gaps_.append(gap)
        except KeyboardInterrupt:
            break

        
    if SAVE:
        data.save(gaps_, NAME)
        print('Data Saved.')




xs = np.array(range(start, end, 2))
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

figure.plot(ln_xs, ln_gaps, ax, labels, line=[SLOPE, b])

if SAVE:
    fig.savefig(f"figures/lnln_{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

plt.show()