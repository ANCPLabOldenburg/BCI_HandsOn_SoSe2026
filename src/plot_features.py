import numpy as np
import matplotlib.pyplot as plt


def plot_features(feat_vec, sel_chan, n_freq):
    """
    Plot feature space with channels x frequencies

    Parameters:
    -----------
    feat_vec : numpy.ndarray
        Feature vector to be plotted
    sel_chan : list or numpy.ndarray
        Selected channel indices (ecog.selectedChannels)
    n_freq : int
        Number of frequencies in feat_vec
    """

    # Reshape the feature vector
    feat_matr = np.reshape(feat_vec, (len(sel_chan), n_freq), order='F')

    # Create full matrix with all channels (assuming 40 channels total)
    plot_matr = np.zeros((40, n_freq))
    plot_matr[np.array(sel_chan) - 1, :] = feat_matr

    # Create the plot
    plt.figure(figsize=(10, 6))
    img = plt.imshow(plot_matr, aspect='auto')

    # Set labels and ticks
    plt.xlabel('Frequencies', fontsize=18)
    plt.ylabel('Channels/Electrodes', fontsize=18)

    # Set x-axis ticks (MATLAB example had 8 ticks from 10 to 80)
    xticks = np.linspace(0, n_freq - 1, 8)  # 8 evenly spaced ticks
    plt.xticks(xticks, ['20', '40', '60', '80', '100', '120', '140', '160'])

    # Add colorbar
    plt.colorbar(img)
    plt.tight_layout()
    plt.show()