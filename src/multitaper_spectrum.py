import os
import pickle
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import dpss, hann, hamming, blackman, flattop, barthann
from scipy.interpolate import interp1d


def multitaper_spectrum(ecog, params):
    """
    Compute multitaper power spectrum for ECoG data.

    Parameters:
    -----------
    ecog : dict
        ECoG data structure at least containing:
        - 'data': numpy array of shape (n_trials/channels, n_samples)
        - 'sampDur': sampling duration in ms
    params : dict
        Parameters for spectral analysis containing:
        - 'tapers': [TW, K] where TW is time-bandwidth product and K is number of tapers
        - 'pad': padding factor
        - 'Fs': sampling frequency (optional, will be calculated from sampDur if not provided)
        - 'fpass': [fmin, fmax] frequency range of interest
        - 'err': error calculation (not implemented)
        - 'trialavg': boolean, whether to average across trials/channels

    Returns:
    --------
    f : numpy array
        Frequency vector
    S : numpy array
        Power spectrum (averaged according to params)

    <Last updated: XW 05.04.25, scipy==1.15.2>
    """

    # Check frequency range doesn't exceed Nyquist
    if params['fpass'][-1] > params['Fs'] / 2:
        print(f"Requested upper frequency ({params['fpass'][-1]} Hz) exceeds Nyquist frequency. "
              f"Correcting to Nyquist ({params['Fs'] / 2} Hz).")
        fmax = params['Fs'] / 2
        params['fpass'] = [0, fmax]

    # Extract taper parameters
    if len(params['tapers']) == 2:
        TW, K = params['tapers']    # TW(time-bandwidth product), K(number of tapers)


    """Approach 1: oringinal approch(works on Linux, but not on Win, because of M length)"""
    # tapers = dpss(len(ecog['data'][0]), NW=TW, Kmax=K)

    """Approach 2: Different windows approach
    test then: np.sum(tapers[0]*tapers[1]) <= 1e-9. Which failed the test 
    ==> maybe not a good approach
    """
    # # 1. Generate diff. windows(not orthogonal)
    # window_funcs = [hann, hamming, blackman, flattop, barthann]
    # assert K <= len(window_funcs), f"K={K} exceeds available windows ({len(window_funcs)})"
    # tapers = np.zeros((len(ecog['data'][0]), K))  # (n_samples, K)
    # for k in range(K):
    #     tapers[:, k] = window_funcs[k](len(ecog['data'][0]))  # each taper has diff. window
    # # 2. QR Decomposition
    # tapers, _ = np.linalg.qr(tapers)  # for Orthogonality

    """ Approach 3: Interpolation (function 'get_long_dpss' designed)
    Problem for win-users by calling scipy.signal.windows.dpss, due to long signals. Orthogonal windows can not be 
    correctly calculated. According to the documentation of function 'dpss': 
        For very long signals (e.g., 1e6 elements), it can be useful to compute
        windows orders of magnitude shorter and use interpolation (e.g.,
        `scipy.interpolate.interp1d`) to obtain tapers of length `M`,
        but this in general will not preserve orthogonality between the tapers.
    Problem of this approach: windows are not guaranteed to be orthogonal to each other
    ==> using QR decomposition to force the tapers to be orthogonal
    """

    # Generate tapers using interpolation approach
    def get_long_dpss(M, NW, Kmax, interp_factor=100):
        short_M = int(M // interp_factor)
        short_tapers = dpss(short_M, NW, Kmax)

        x_short = np.linspace(0, 1, short_M)
        x_long = np.linspace(0, 1, M)
        tapers = np.zeros((Kmax, M))
        for k in range(Kmax):
            interp_func = interp1d(x_short, short_tapers[k], kind='cubic')
            tapers[k] = interp_func(x_long)
        return tapers

    # calculate
    N_samples = ecog['data'].shape[1]  # 254

    # Apply interpolation, only if the number of samples is large (1e6)
    if N_samples < 100000:
        tapers = dpss(N_samples, NW=TW, Kmax=K)
    else:
        tapers = get_long_dpss(N_samples, TW, K)

    # tapers = get_long_dpss(len(ecog['data'][0]), TW, K)

    # Option: forced to be orthogonal
    tapers, _ = np.linalg.qr(tapers.T)
    tapers = tapers.T

    # Scale tapers (MATLAB source: dpsschk.m: tapers = tapers*sqrt(Fs);)
    tapers = tapers.T * np.sqrt(params['Fs'])

    # Set up FFT parameters
    nfft = len(ecog['data'][0])

    # Frequency grid
    freqs = fftfreq(nfft, 1 / params['Fs'])
    findx = np.where((freqs >= params['fpass'][0]) & (freqs <= params['fpass'][1]))[0]
    f = freqs[findx]

    # Compute multitaper spectrum
    J = np.zeros((len(findx), tapers.shape[1], len(ecog['data'])), dtype=complex)
    for i in range(len(ecog['data'])):
        for j in range(tapers.shape[1]):
            x = ecog['data'][i, :] * tapers[:, j]
            X = fft(x, n=nfft) / params['Fs']
            J[:, j, i] = X[findx]

    S = np.squeeze(np.mean(np.conj(J) * J, axis=1))
    # S = np.squeeze(np.mean(np.abs(J) ** 2, axis=1))

    # Average over trials/channels if required
    if params['trialavg']:
        S = np.mean(S, axis=1)

    return f, S