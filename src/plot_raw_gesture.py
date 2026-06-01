import numpy as np
import matplotlib.pyplot as plt


def plot_raw_gesture(gloveResamp):
    """
    Plot raw glove sensor data including finger bending, pitch, roll, and accelerometer values.

    Parameters:
    gloveResamp - Dictionary containing glove sensor data
    """
    # Generate time axis (MATLAB equivalent: linspace)
    timebase = np.linspace(1, len(gloveResamp['timebase']) / 200, len(gloveResamp['timebase']))

    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # **First subplot: Gyroscope & Fingerbend sensors**
    axs[0].plot(timebase, gloveResamp['fingers'][1], 'k', linewidth=2, label='Hand/Fist')
    axs[0].plot(timebase, gloveResamp['fingers'][3], 'b', linewidth=2, label='Finger')
    axs[0].plot(timebase, gloveResamp['pitch'], 'r', linewidth=2, label='Pitch')
    axs[0].plot(timebase, gloveResamp['roll'], 'g', linewidth=2, label='Roll')

    axs[0].set_xlim([10, 85])
    axs[0].set_ylabel('Amplitude []', fontweight='bold', fontsize=14)
    axs[0].legend()
    axs[0].set_title('Gyroscope and Fingerbend Sensors', fontsize=14)

    # **Second subplot: Accelerometer data**
    axs[1].plot(timebase, gloveResamp['acceleratorsXYZ'][0], 'r', linewidth=2, label='X')
    axs[1].plot(timebase, gloveResamp['acceleratorsXYZ'][1], 'b', linewidth=2, label='Y')
    axs[1].plot(timebase, gloveResamp['acceleratorsXYZ'][2], 'g', linewidth=2, label='Z')

    axs[1].set_xlim([10, 85])
    axs[1].set_xlabel('Time [s]', fontweight='bold', fontsize=14)
    axs[1].set_ylabel('Amplitude [g]', fontweight='bold', fontsize=14)
    axs[1].legend()
    axs[1].set_title('Accelerometers', fontsize=14)

    plt.tight_layout()
    plt.show()


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
