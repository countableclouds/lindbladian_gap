import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import filter
import graph
from graph import HypercubeGraph, EigResult
import data
from lindbladian import Lindbladian, sparsify_jump
import figure
from scipy.linalg import block_diag
import random


beta = 1
gaps_ = []

GRAPH= 'cyclic'
FILTER = 'ckg_metropolis'
SLOPE = -3
NAME = f"{FILTER}_{GRAPH}"


LOAD = False
SAVE = False

if LOAD:
    gaps_ = data.load(NAME)
    
# 1/n^0 => lambda = 1/n^2
# 1/n^(1/3) => lambda = 1/n
# 1/n^(1/2) => lambda = 1/sqrt(n)
# 1/n^1 => lambda = 1

#Start = 4 and End = 50 for most of them, start = 1 and end = something for hypercube (fix this)
start = 4
end = 30
pbar = tqdm(total=end-start)

if LOAD:
    print('Loaded.')
else:
    for n in range(start, end):
        F = filter.Filter.from_name(FILTER)(1)
        start_time = time.time()
        
        M = graph.CyclicGraph.adj_matrix(n)

        jumps = []
        # nonzero = np.nonzero(M)
        # for i in range(len(nonzero[0])):
        #     N= np.zeros((n, n))
        #     N[nonzero[0][i], nonzero[1][i]] = 2**(-0.5)
        #     jumps.append(sparsify_jump(N))
        
        eigenvalues = graph.CyclicGraph.energies(n)
        eigenvectors = graph.CyclicGraph.eigenvectors(n)
    # print(eigenvalues, eigenvectors)
        eigenvector_preset = graph.EigResult(eigenvalues = eigenvalues, 
                                        eigenvectors = eigenvectors)

        # for i in range(n):
        #     N= np.zeros((n, n))
        #     N[i, i] = 1
        #     jumps.append(sparsify_jump(N))
            
        # N= np.zeros((n, n))
        # for i in range(0, n):
        #     N[i, i] += random.randint(-1, 1)
        # jumps.append(sparsify_jump(N))
        
        # N= np.zeros((n, n))
        # for i in range(0, n):
        #     N[i, i] += random.randint(-1, 1)
        # jumps.append(sparsify_jump(N))
        
        N= np.zeros((n, n))
        for i in range(0, n):
            N[i, i] += (-1)**i
        jumps.append(sparsify_jump(N))
        
        N= np.zeros((n, n))
        for i in range(n//2, n):
            N[i, i] += 1
        jumps.append(sparsify_jump(N))
        
        N= np.zeros((n, n))
        for i in range(n//2):
            N[i, i] += 1
        jumps.append(sparsify_jump(N))
        
        N= np.zeros((n, n))
        for i in range(0, n, 2):
            N[i, i] += 1
        jumps.append(sparsify_jump(N))
            
        N= np.zeros((n, n))
        for i in range(1, n, 2):
            N[i, i] += 1
        jumps.append(sparsify_jump(N))

        G = graph.Graph.from_adjacency(M)(jumps, eig =  eigenvector_preset)
        # G = graph.Graph.from_name('cyclic')(n, 'diagonal')
        L = Lindbladian(G, F)
        L.initialize()
        L.cyclic_reshape()
        
        # block_gaps = []
        # for k in range(2):
        #     indices, _ = np.indices((n, 2))
        #     indices[:, 1] = (indices[:, 1]+k)%n
        #     M,D =L.block(indices)

        #     # print(D)
        #     if k ==0:
        #         block_gaps.append(Lindbladian.mat_spectral_gap(-D, 1))
        #     else:
        #         block_gaps.append(Lindbladian.mat_spectral_gap(-D, 0))
        # gaps_.append(min(block_gaps))
        # print(np.argmin(gaps_))
        
        gaps_.append(L.spectral_gap())
        # print(gaps_)
        elapsed_time = time.time() - start_time
        pbar.set_postfix({"Elapsed Time": f"{elapsed_time:.2f}s"})
        pbar.update(1)
        # assert 0==1
        
        
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