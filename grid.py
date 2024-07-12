import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import filter
import graph
from matplotlib import rc
import data
import lindbladian
import figure


beta = 1
gaps_ = []

GRAPHS= ['cyclic', 'path', 'complete']
FILTER = 'davies_glauber'
SLOPES = [0, 0, -1]
NAMES = [f"{FILTER}_{GRAPH}" for GRAPH in GRAPHS]

F = filter.Filter.from_name(FILTER)(1)

gaps = [np.array(data.load(NAME)) for NAME in NAMES]

xs = range(4, len(gaps[0]) + 4)

ln_xs = np.log(xs)
ln_gaps = [np.log(elem) for elem in gaps]

bs = [np.mean(ln_gap[-10:] - SLOPES[i]*ln_xs[-10:]) for i, ln_gap in enumerate(ln_gaps)]

fig, axs = plt.subplots(2, 3, figsize=(15, 6))
fig.tight_layout()
labels = ['Size of Graph (n)', 'Spectral Gap']
axs[0, 0].set_title('Cyclic (Θ(n⁻²))', fontsize = 20)
figure.plot(xs, gaps[0], axs[0, 0], labels)
axs[0, 1].set_title('Path (Θ(n⁻²))', fontsize = 20)
figure.plot(xs, gaps[1], axs[0, 1], labels)
axs[0, 2].set_title('Complete (Θ(n⁻¹))', fontsize = 20)
figure.plot(xs, gaps[2], axs[0, 2], labels)

labels = ['Log Size of Graph (ln n)', 'Log Spectral Gap']
for i in range(3):
    if SLOPES[i]:
        figure.plot(ln_xs, ln_gaps[i], axs[1, i], labels, line = [SLOPES[i], bs[i]])
    else:
        figure.plot(ln_xs, ln_gaps[i], axs[1, i], labels)
fig.subplots_adjust(hspace=0.3)
fig.subplots_adjust(wspace=0.3)

fig.savefig(f"figures/grid_davies.pdf", bbox_inches='tight')

# plt.show()