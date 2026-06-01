import numpy as np

def remove_bad_epochs(epoch, selected_intervals_in_gui_units):
    """
    Removes bad epochs from the epoch structure.

    Parameters:
    -----------
    epoch : dict
        Epoch structure containing information about each trial (onset, label, duration).
    selected_intervals_in_gui_units : list or numpy.ndarray
        Manually defined bad intervals in seconds.

    Returns:
    --------
    epoch : dict
        Updated epoch structure with bad epochs removed.
    """

    # Length of each epoch in seconds
    len_interval = 0.25  # 250 ms epochs

    # Convert bad intervals from seconds to sample indices
    selected_intervals_in_gui_units = np.round(np.array(selected_intervals_in_gui_units) * epoch['srate']).astype(int)

    # Initialize list to store bad epoch indices
    bad_epoch = []

    # Find the first bad interval that lies within the epoch range
    start_idx = np.where(selected_intervals_in_gui_units[:, 0] > epoch['OnsetIdx'][0])[0]
    if len(start_idx) > 0:
        start_idx = start_idx[0]
    else:
        start_idx = 0

    # Iterate over each bad interval
    for i in range(start_idx, len(selected_intervals_in_gui_units)):
        # Find the first epoch that lies within the bad interval
        bad_epoch_on = np.where(epoch['OnsetIdx'] > selected_intervals_in_gui_units[i, 0])[0]
        if len(bad_epoch_on) > 0:
            bad_epoch_on = bad_epoch_on[0]
            # Check if the previous epoch also lies within the bad interval
            if (epoch['OnsetIdx'][bad_epoch_on - 1] + int(epoch['srate'] * len_interval)) > selected_intervals_in_gui_units[i, 0]:
                bad_epoch_on -= 1
        else:
            continue

        # Find the last epoch that lies within the bad interval
        bad_epoch_off = np.where(epoch['OnsetIdx'] > selected_intervals_in_gui_units[i, 1])[0]
        if len(bad_epoch_off) > 0:
            bad_epoch_off = bad_epoch_off[0] - 1
        else:
            bad_epoch_off = len(epoch['OnsetIdx']) - 1

        # If the bad interval contains valid epochs, add them to the bad_epoch list
        if bad_epoch_off >= bad_epoch_on:
            bad_epoch.extend(range(bad_epoch_on, bad_epoch_off + 1))

    # Remove bad epochs from the epoch structure
    if bad_epoch:
        epoch['OnsetIdx'] = np.delete(epoch['OnsetIdx'], bad_epoch)
        epoch['label'] = np.delete(epoch['label'], bad_epoch)

    return epoch