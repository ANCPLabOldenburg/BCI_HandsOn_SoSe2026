import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

# Ensure correct Matplotlib backend for interactive mode
matplotlib.use("Qt5Agg")


def _safe_ginput(fig, ax, n, clear_on_exit=True):
    """
    Collect n left-click points from the axes, ignoring clicks made while
    a navigation tool (zoom / pan) is active in the toolbar.

    This is a drop-in replacement for plt.ginput(n, timeout=0) that prevents
    zoom-box or pan-drag clicks from being accidentally recorded as data points.

    Parameters
    ----------
    fig          : matplotlib.figure.Figure
    ax           : matplotlib.axes.Axes - only clicks inside this axes are accepted
    n            : int                  - number of points to collect
    clear_on_exit: bool (default True)  - if True, remove this round's markers from
                   the axes before returning; if False, leave them visible so the
                   caller can manage them (e.g. clear them at the start of the next
                   round after inspecting them).

    Returns
    -------
    points  : list of (x, y) tuples (same format as plt.ginput)
    markers : list of Line2D objects for the plotted × markers
    """
    points = []
    markers = []

    def on_click(event):
        if event.inaxes is not ax:
            return
        # Ignore clicks while a navigation tool (zoom rect / pan) is active
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != "":
            return
        if event.button == 1 and event.xdata is not None:  # left-click with valid coords
            points.append((event.xdata, event.ydata))
            marker, = ax.plot(event.xdata, event.ydata, 'rx', markersize=12, markeredgewidth=2)
            markers.append(marker)
            fig.canvas.draw()
            if len(points) >= n:
                fig.canvas.stop_event_loop()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.start_event_loop(timeout=0)  # blocks until stop_event_loop() is called
    fig.canvas.mpl_disconnect(cid)

    if clear_on_exit:
        for m in markers:
            m.remove()
        if markers:
            fig.canvas.draw()

    return points, markers


def get_epochs(glove_data, analog_data, len_interval, keep_markers=False):
    """
    Interactive function to get movement onset timestamps (supports multiple gestures).

    Parameters
    ----------
    keep_markers : bool (default False)
        If False (default), the red × markers from the previous round are removed
        from the plot before the next round starts.
        If True, all markers remain visible across rounds so you can compare
        selections visually.
    """
    # Rescale analog data
    analog_data_round = np.round(np.array(analog_data) * 10)

    # Generate time axis
    timebase1 = np.arange(len(glove_data['gesture']))

    # Time shift adjustment (manual tuning)
    shift_block1 = -41886

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(analog_data_round, label="Analog", linestyle="-")
    ax.plot(timebase1 + shift_block1, glove_data['gesture'], '--r', label="Gesture")
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][1], 'k', label="Hand/Fist")
    ax.plot(timebase1 + shift_block1, glove_data['fingers'][3], 'b', label="Finger")
    ax.plot(timebase1 + shift_block1, glove_data['pitch'], 'r', label="Pitch")
    ax.plot(timebase1 + shift_block1, glove_data['roll'], 'g', label="Roll")

    ax.set_xlim([0.8e5, 1.6e5])
    ax.set_title("Glove Data Timeseries with Movement Onsets", fontweight="bold")
    ax.set_ylabel("Amplitude", fontweight="bold")
    ax.legend()

    # Show interactive figure and keep the window open
    plt.show(block=False)

    # Interactively get multiple gestures
    exec_flag = "y"
    finger = []
    prev_markers = []  # markers from the previous round
    all_markers = []   # all markers ever drawn (used when keep_markers=True)

    while exec_flag.lower() == "y":
        # Clear markers left over from the previous round (only when keep_markers is off)
        if not keep_markers:
            for m in prev_markers:
                m.remove()
            if prev_markers:
                fig.canvas.draw()
        prev_markers = []

        print("[OK] Use zoom/pan tools to navigate, then deactivate them.")
        print("[OK] Click 3 points on the plot (start, middle, end).")
        # keep_on_exit=False so markers stay visible until the *next* round begins
        points, prev_markers = _safe_ginput(fig, ax, n=3, clear_on_exit=False)
        all_markers.extend(prev_markers)

        if len(points) == 3:
            finger.append([p[0] for p in points])
        else:
            ...
            #print("⚠️ You did not select exactly 3 points! Please try again.")

        exec_flag = input("Do you want to record another gesture? [y/n]: ")

    # Clear any remaining markers before closing
    for m in (all_markers if keep_markers else prev_markers):
        m.remove()
    plt.close()  # Close plot window

    # Process `finger` data
    finger = np.array(finger).T  # Convert to (3, N) structure
    print(f"[OK] Recorded {finger.shape[1]} gesture events.")

    # Sampling rate
    srate = glove_data['srate']
    epoch = {"OnsetIdx": [], "label": []}

    # Compute gesture start (onset) and end (offset) times
    finger_up = np.round(finger[:2, :])
    finger_down = np.round(finger[1:3, :])

    # Keep only intervals larger than `len_interval`
    valid_up = (finger_up[1, :] - finger_up[0, :]) >= round(srate * len_interval)
    valid_down = (finger_down[1, :] - finger_down[0, :]) >= round(srate * len_interval)

    finger_up = finger_up[:, valid_up]
    finger_down = finger_down[:, valid_down]

    nb_finger_up = np.floor((finger_up[1, :] - finger_up[0, :]) / (srate * len_interval))
    nb_finger_down = np.floor((finger_down[1, :] - finger_down[0, :]) / (srate * len_interval))

    m_finger_up = np.mean(finger_up, axis=0)
    m_finger_down = np.mean(finger_down, axis=0)

    # Compute timestamps for events
    for i in range(finger_up.shape[1]):
        x = np.arange(nb_finger_up[i])
        f_onset = m_finger_up[i] - (nb_finger_up[i] / 2) * round(srate * len_interval)
        epoch["OnsetIdx"].extend((x * round(srate * len_interval) + f_onset).astype(int))
        epoch["label"].extend([20] * len(x))

    for i in range(finger_down.shape[1]):
        x = np.arange(nb_finger_down[i])
        f_onset = m_finger_down[i] - (nb_finger_down[i] / 2) * round(srate * len_interval)
        epoch["OnsetIdx"].extend((x * round(srate * len_interval) + f_onset).astype(int))
        epoch["label"].extend([21] * len(x))

    print("[OK] All movement events successfully recorded!")
    return epoch

def main():
    parser = argparse.ArgumentParser(
        description="Interactive gesture epoch labeling tool for BCI data."
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/raw/data_for_own_epoch.pkl",
        help="Path to input .pkl file containing gloveResamp and ecogAnalog (default: data_for_own_epoch.pkl)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results/epochs.pkl",
        help="Path to output .pkl file for saving epoch data (default: epochs.pkl)"
    )
    parser.add_argument(
        "--epoch-length", "-l",
        type=float,
        default=0.25,
        help="Desired epoch length in seconds (default: 0.25)"
    )
    parser.add_argument(
        "--keep-markers", "-k",
        action="store_true",
        default=False,
        help="Keep red × markers from previous rounds visible on the plot (default: clear them before each new round)"
    )
    args = parser.parse_args()

    # Load data from file
    with open(args.input, "rb") as f:
        data = pickle.load(f)
        gloveResamp = data["gloveResamp"]
        ecogAnalog = data["ecogAnalog"]

    # Run interactive gesture selection
    ownEpochs = get_epochs(gloveResamp, ecogAnalog, args.epoch_length, keep_markers=args.keep_markers)

    # Save epoch data
    with open(args.output, "wb") as f:
        pickle.dump(ownEpochs, f)

    print(f"[OK] Gesture event data saved to `{args.output}`.")


if __name__ == "__main__":
    main()
