import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nearly import nearly
from matplotlib.widgets import Button


def run_bad_channel_removal(input_file, output_file):
    """Run interactive bad channel removal using periodogram visualization."""

    # Load ECoG data
    with open(input_file, 'rb') as f:
        ecog = pickle.load(f)
    ecog['data'] = np.array(ecog['data'])

    # Extract periodogram data
    periodogram = np.array(ecog['periodogram']['periodogram']).T
    centerFrequency = np.array(ecog['periodogram']['centerFrequency'])
    periodogramFrequencyband = ecog['periodogram']['params']['fpass']
    showFrequencyBandInSingleChannelDisplay = np.array([1, 200])

    # Compute power spectrum
    p = None
    for k in range(periodogramFrequencyband[0], periodogramFrequencyband[1]):
        tem = periodogram[:, nearly(k, centerFrequency):(nearly(k + 1, centerFrequency) + 1)]
        tem_avg = np.mean(tem, axis=1)
        p = tem_avg if p is None else np.vstack((p, tem_avg))
    p = p.T

    # Frequency range
    f = np.array(range(periodogramFrequencyband[0], periodogramFrequencyband[1]))
    idx = nearly(showFrequencyBandInSingleChannelDisplay, f)
    f = f[idx[0]:idx[1] + 1]

    # Compute log-transformed mean & std
    selected_channels = np.array(ecog['selectedChannels']) - 1
    log_p = np.log10(p[selected_channels, idx[0]:idx[1] + 1])
    s = np.std(log_p, axis=0)
    m = np.mean(log_p, axis=0)

    # Store bad channels
    badChannels = []

    # Track current channel
    state = {'current_channel': 0}

    # Define the callback functions
    def mark_good(event):
        """Mark the channel as good and move to next"""
        plt.close()
        state['current_channel'] += 1
        if state['current_channel'] < periodogram.shape[0]:
            plot_channel(state['current_channel'])

    def mark_bad(event):
        """Mark the channel as bad and move to next"""
        badChannels.append(state['current_channel'] + 1)  # Store as 1-based index (MATLAB convention)
        plt.close()
        state['current_channel'] += 1
        if state['current_channel'] < periodogram.shape[0]:
            plot_channel(state['current_channel'])

    # Function to plot the current channel
    def plot_channel(k):
        """Plot power spectrum for channel k with interactive buttons"""
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(bottom=0.2)

        # Plot mean +/- std
        ax.plot(f, m - s, 'k-', label='Mean - 1 STD')
        ax.plot(f, m + s, 'k-', label='Mean + 1 STD')
        ax.plot(f, np.log10(p[k, idx[0]:idx[1] + 1]), 'r', label=f'Channel {k + 1}')

        ax.set_title(f'Channel #: {k + 1}')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Log Energy [log10 uV]')
        ax.legend()

        # Define button positions
        ax_good = plt.axes([0.3, 0.035, 0.15, 0.075])
        ax_bad = plt.axes([0.55, 0.035, 0.15, 0.075])

        # Create buttons
        btn_good = Button(ax_good, 'Good (Y)')
        btn_bad = Button(ax_bad, 'Bad (N)')

        btn_good.on_clicked(mark_good)
        btn_bad.on_clicked(mark_bad)

        # Keyboard shortcuts: Y → Good, N → Bad
        def on_key(event):
            if event.key in ('y'):
                mark_good(event)
            elif event.key in ('n'):
                mark_bad(event)

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    # Start the first plot
    plot_channel(state['current_channel'])

    # Print the final list of bad channels after all plots are closed
    print("Bad channels:", badChannels)
    ecog['badChannels'] = badChannels

    with open(output_file, "wb") as file:
        pickle.dump(ecog, file)

    print(f"[OK] Results saved to `{output_file}`.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive bad channel removal tool using periodogram visualization."
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw/ecogStruct1_processed.pkl",
        help="Path to input .pkl file containing ECoG data (default: ecogStruct1_processed.pkl)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/raw/ecogStruct1_periodogram.pkl",
        help="Path to output .pkl file for saving results (default: ecogStruct1_periodogram.pkl)"
    )
    args = parser.parse_args()
    run_bad_channel_removal(args.input, args.output)


if __name__ == "__main__":
    main()