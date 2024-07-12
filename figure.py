import matplotlib.pyplot as plt

def plot(xs, gaps, ax, labels, line = None):
    xlabel, ylabel = labels
    
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    ax.plot(xs, gaps, marker='o', linestyle='None', markersize=6, markerfacecolor='blue', markeredgewidth=1, markeredgecolor='black')
    if line:
        ax.plot([xs[0], xs[-1]], [xs[0]*line[0]+line[1], xs[-1]*line[0]+line[1]], linestyle='-', color='green', linewidth=2, label = f"Slope: {line[0]}") 
        ax.legend(fontsize=14,loc='upper right', bbox_to_anchor=(1, 0.97))

    