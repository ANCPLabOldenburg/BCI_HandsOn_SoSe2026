import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Ensure Matplotlib runs in interactive window mode
import matplotlib.pyplot as plt

def plot_own_labels(glove_data, analog_data, own_epochs):
    """
    Plot glove data with user-defined movement labels.

    Parameters:
    glove_data  - Dictionary containing glove sensor data
    analog_data - Array of analog data values
    own_epochs  - Dictionary containing user-defined movement onset and offset indices
    """

    # Rescale analog data
    analog_data_round = np.round(np.array(analog_data) * 10)

    # Generate time axis
    timebase1 = np.arange(len(glove_data['gesture']))

    # Adjust time offset
    shift_block1 = -41886

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(analog_data_round, label="Analog", linestyle="-")
    ax.plot(timebase1 + shift_block1, glove_data['gesture'], '--r', label="Gesture")
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][1], 'k', label="Hand/Fist")
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][3], 'b', label="Finger")
    ax.plot(timebase1 + shift_block1, glove_data['pitch'], 'r', label="Pitch")
    ax.plot(timebase1 + shift_block1, glove_data['roll'], 'g', label="Roll")

    ax.set_xlim([0.8e5, 1.6e5])
    ax.set_title("Glove Data Timeseries with User-defined Labels", fontweight="bold")
    ax.set_ylabel("Amplitude", fontweight="bold")
    ax.legend()

    # Plot movement onset/offset markers
    for i in range(len(own_epochs["label"])):
        onset_idx = own_epochs["OnsetIdx"][i]
        label = own_epochs["label"][i]

        if label == 20:  # Movement onset
            ax.axvline(x=onset_idx, color='b', linestyle='-')
        elif label == 21:  # Movement offset
            ax.axvline(x=onset_idx, color='b', linestyle=':')

    plt.show()
