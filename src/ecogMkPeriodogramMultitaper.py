import numpy as np
from scipy.signal import spectrogram

def ecogMkPeriodogramMultitaper(ecog, trialList=None, params=None):
    """
    Calculate a periodogram for each data channel using multitaper method.

    Parameters:
    -----------
    ecog : dict
        An ECoG structure with at least the fields 'data' and 'sampDur'.
    trialList : list, optional
        A list of trials to be included. If None, all trials are included.
    params : dict, optional
        A dictionary holding parameters such as the range of frequencies, etc.
        Default values will be used for omitted parameters.

    Returns:
    --------
    ecog : dict
        The ECoG structure with the field 'periodogram' added.
    """
    # Default parameters
    if params is None:
        params = {
            'Fs': 1000 / ecog['sampDur'],  # Sampling frequency
            'fpass': [0, (1000 / ecog['sampDur']) / 2],  # Frequency range of interest
            'tapers': [3, 5],  # Tapers
            'trialave': 0,  # Average over trials (0 = no, 1 = yes)
            'err': 0,  # Error computation (0 = no, 1 = yes)
            'pad': -1  # Padding for frequency analysis
        }

    # Validate frequency range
    if params['fpass'][1] > params['Fs'] / 2:
        print(f"Requested upper frequency ({params['fpass'][1]} Hz) exceeds Nyquist frequency. "
              f"Correcting to Nyquist ({params['Fs'] / 2} Hz).")
        params['fpass'][1] = params['Fs'] / 2

    # Initialize periodogram structure
    ecog['periodogram'] = {
        'trialList': trialList if trialList is not None else list(range(ecog['data'].shape[2])),
        'params': params,
        'centerFrequency': None,
        'periodogram': None
    }

    # Calculate periodograms for each trial
    for k in ecog['periodogram']['trialList']:
        tmp = ecog['data'][:, :, k]  # Extract data for the current trial

        # Compute spectrogram using scipy.signal.spectrogram
        f, t, Sxx = spectrogram(
            tmp.T,  # Input data (transposed to match scipy's format)
            fs=params['Fs'],  # Sampling frequency
            nperseg=params['tapers'][0] * 2,  # Segment length
            noverlap=params['tapers'][0],  # Overlap
            nfft=params['pad'] if params['pad'] != -1 else None  # Padding
        )

        # Filter frequencies within the specified range
        freq_mask = (f >= params['fpass'][0]) & (f <= params['fpass'][1])
        f = f[freq_mask]
        Sxx = Sxx[freq_mask, :]

        # Initialize periodogram array on the first iteration
        if k == ecog['periodogram']['trialList'][0]:
            ecog['periodogram']['periodogram'] = np.zeros((Sxx.shape[0], Sxx.shape[1], len(ecog['periodogram']['trialList'])))
            ecog['periodogram']['centerFrequency'] = f

        # Store the periodogram for the current trial
        ecog['periodogram']['periodogram'][:, :, k] = Sxx

    # Average over trials if requested
    if params['trialave'] == 1:
        ecog['periodogram']['periodogram'] = np.mean(ecog['periodogram']['periodogram'], axis=2)

    return ecog