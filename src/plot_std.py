import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Matplotlib runs in interactive window mode
import matplotlib.pyplot as plt

def plot_std(ecog, s):

    badChannels = np.array(ecog['badChannels']) - 1
    idxChannels = np.array(np.arange(1, len(s) + 1))

    # Plot the standard deviation
    plt.ion()
    plt.figure()
    plt.plot(idxChannels, s, label='Standard Deviation')
    plt.axis('tight')
    plt.xticks(np.linspace(1, len(idxChannels), num=9, dtype=int))

    # Mark the bad channels
    for bc in idxChannels[badChannels]:
        plt.plot(bc, s[bc - 1], 'r.', markersize=10, label='Currently Marked Bad Channels')
        plt.axvline(x=bc, linestyle='--', color='r', alpha=0.6)

    # Add plot information
    plt.legend(['Standard Deviation', 'Currently Marked Bad Channels'])

    plt.show(block=True)
