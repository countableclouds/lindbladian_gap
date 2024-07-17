import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from filter import Filter
from graph import HypercubeGraph, Graph, EigResult
import data
from lindbladian import Lindbladian
import figure
from scipy.linalg import block_diag



gaps_ = []

GRAPH= 'custom'
FILTER = 'ckg_metropolis'
LOGLOG_SLOPE = -1
NAME = f"{FILTER}_{GRAPH}"


LOAD = False
SAVE = False

if LOAD:
    gaps_ = data.load(NAME)

#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 2
end = 8
pbar = tqdm(total=end-start)

if LOAD:
    print('Loaded.')
else:
    for d in range(start, end):
        beta = 1
        F = Filter.from_name(FILTER)(beta)
        n = 2**d
        start_time = time.time()
        
        M = HypercubeGraph.adj_matrix(d)
        eigenvalues = HypercubeGraph.energies(d)
        eigenvectors = HypercubeGraph.eigenvectors(d)
        eigenvector_preset = EigResult(eigenvalues = eigenvalues, 
                                       eigenvectors = eigenvectors)
        N= np.identity(n)
        N = M
        G = Graph.from_name(GRAPH, adj = M, eig = eigenvector_preset)(N)
        # G = Graph.from_name('hypercube')(d, 'diagonal')
        # print(eigenvectors)
        # print(G.energies)
        L = Lindbladian(G, F)
        
        block_gaps = []
        for k in range(2):
            indices, _ = np.indices((n, 2))
            indices[:, 1] = (indices[:, 1]^k)%n
            M,D =L.block(indices)
            # print(D)
            if k ==0:
                block_gaps.append(Lindbladian.mat_spectral_gap(-D, 1))
            else:
                block_gaps.append(Lindbladian.mat_spectral_gap(-D, 0))
        
        gaps_.append(min(block_gaps))
        elapsed_time = time.time() - start_time
        pbar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f}s"})
        pbar.update(1)
        
        
    if SAVE:
        data.save(gaps_, NAME)
        print('Data Saved.')

pbar.close()
LOG_SLOPE = np.log((np.exp(beta) + np.exp(-beta))/(2 * np.exp(beta)))
xs = np.array(range(start, end))
gaps = np.array(gaps_)
print(gaps)
ln_xs = np.log(xs)
ln_gaps = np.log(gaps)
b = np.mean(ln_gaps[-3:] - LOG_SLOPE*xs[-3:])

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (n)', 'Spectral Gap']
figure.plot(xs, gaps, ax, labels)
if SAVE:
    fig.savefig(f"figures/{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Size of Graph (ln n)', 'Log Spectral Gap']
if LOG_SLOPE:
    figure.plot(xs, ln_gaps, ax, labels, line=[LOG_SLOPE, b])
else:
    figure.plot(xs, ln_gaps, ax, labels)
if SAVE:
    fig.savefig(f"figures/ln_{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

fig, ax = plt.subplots(figsize = (9, 7))
labels = ['Log Size of Graph (ln n)', 'Log Spectral Gap']
if LOGLOG_SLOPE:
    figure.plot(ln_xs, ln_gaps, ax, labels, line=[LOGLOG_SLOPE, b])
else:
    figure.plot(ln_xs, ln_gaps, ax, labels)
if SAVE:
    fig.savefig(f"figures/lnln_{NAME}.pdf", bbox_inches='tight', pad_inches=0.25)

plt.show()

plt.show()
