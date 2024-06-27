import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import filter
import graph
import lindbladian


beta=1
gaps_ = []
F = filter.MetropolisFilter(1)
for n in tqdm(range(4, 50)):
    G = graph.CyclicGraph(n, 'diagonal') 
    L = lindbladian.Lindbladian(G, F)
    gaps_.append(L.spectral_gap(assertion=True))

xs =range(4, len(gaps_)+4)
    

gaps = np.array(gaps_)
plt.title("Cyclic Graph (beta=1) with Metropolis Filter (Davies Generator)")
plt.xlabel("Size of Graph (n)")
plt.ylabel("Spectral Gap")
plt.plot(xs, gaps)
plt.show()