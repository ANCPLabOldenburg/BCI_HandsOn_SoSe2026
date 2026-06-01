import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Matplotlib runs in interactive window mode
import matplotlib.pyplot as plt


def plot_glove_data(glove_data, analog_data, trigger):
    """
    Plot glove data including analog sensor signals, hand/finger movements, and motion onset triggers.

    Parameters:
    glove_data  - Dictionary containing glove sensor data
    analog_data - Array of analog data values
    trigger     - Dictionary with movement onset and offset indices

    Example Usage:
    plot_glove_data(gloveResamp, ecogAnalog, epoch)
    """

    # Rescale analog data
    analog_data_round = np.round(np.array(analog_data) * 10)

    # Generate time axis (MATLAB equivalent: 1:length(gloveData.gesture{1}))
    timebase1 = np.arange(len(glove_data['gesture']))

    # Adjust time offset for proper alignment
    shift_block1 = -41886

    # Create interactive figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot analog sensor data
    ax.plot(analog_data_round, label="Analog", linestyle="-")

    # Plot gesture movement data
    ax.plot(timebase1 + shift_block1, glove_data['gesture'], '--r', label="Gesture")

    # Plot finger sensor data
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][1], 'k', label="Hand/Fist")
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][3], 'b', label="Finger")

    # Plot motion tracking data (Pitch & Roll)
    ax.plot(timebase1 + shift_block1, glove_data['pitch'], 'r', label="Pitch")
    ax.plot(timebase1 + shift_block1, glove_data['roll'], 'g', label="Roll")

    plt.ion()  # Enable interactive mode

    # Configure legend and labels
    ax.legend()
    ax.set_title("Glove Data Timeseries with Movement Onsets", fontweight="bold")
    ax.set_ylabel("Amplitude", fontweight="bold")

    # Plot movement onset/offset markers
    for i in range(len(trigger['label'])):
        onset_idx = trigger['OnsetIdx'][i]
        if trigger['label'][i] == 20:
            ax.axvline(x=onset_idx, color='b', linestyle='-')
        elif trigger['label'][i] == 21:
            ax.axvline(x=onset_idx, color='b', linestyle=':')

    # Show interactive plot
    plt.show(block=True)