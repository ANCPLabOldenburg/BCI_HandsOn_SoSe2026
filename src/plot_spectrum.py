import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def plot_spectrum(data, fs, data_filtered=None, log_scale=False):
    """
    Plot time series and corresponding amplitude spectrum.

    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data.
    fs : float
        Sampling frequency.
    data_filtered : numpy.ndarray, optional
        Filtered time series data.
    log_scale : bool, optional
        If True, plot the log of the amplitude spectrum.
    """
    # Length of the data
    L = len(data)

    # Next power of 2 for zero-padding
    NFFT = 2 ** int(np.ceil(np.log2(L)))

    # Compute FFT of the original data
    Y = fft(data, NFFT) / L
    f = fftfreq(NFFT, 1 / fs)[:NFFT // 2 + 1]  # Frequency vector (positive frequencies only)

    if data_filtered is None:
        # Plot time series and amplitude spectrum (no filtered data)
        plt.figure(figsize=(12, 8))

        # Time series plot
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.xlabel('Time [ms]', fontsize=24)
        plt.ylabel('Amplitude [mV]', fontsize=24)
        plt.title('Time Series')

        # Amplitude spectrum plot
        plt.subplot(2, 1, 2)
        if log_scale:
            plt.plot(f, np.log10(2 * np.abs(Y[:NFFT // 2 + 1])))
            plt.ylabel('log |Y(f)|', fontsize=24)
        else:
            plt.plot(f, 2 * np.abs(Y[:NFFT // 2 + 1]))
            plt.ylabel('|Y(f)|', fontsize=24)
        plt.xlabel('Frequency [Hz]', fontsize=24)
        plt.title('Amplitude Spectrum')
        plt.xlim(0, fs / 2)  # Ensure frequency starts from 0

    else:
        # Plot time series and amplitude spectrum (with filtered data)
        plt.figure(figsize=(16, 12))

        # Time series plot (original data)
        plt.subplot(2, 2, 1)
        plt.plot(data)
        plt.xlabel('Time [ms]', fontsize=24)
        plt.ylabel('Amplitude [mV]', fontsize=24)
        plt.title('Time Series - Original Data')
        plt.xlim(0, len(data))

        # Amplitude spectrum plot (original data)
        plt.subplot(2, 2, 3)
        if log_scale:
            plt.plot(f[0:-1], np.log10(2 * np.abs(Y[:NFFT // 2 + 1][0:-1])))
            plt.ylabel('log |Y(f)|', fontsize=24)
        else:
            plt.plot(f, 2 * np.abs(Y[:NFFT // 2 + 1]))
            plt.ylabel('|Y(f)|', fontsize=24)
        plt.xlabel('Frequency [Hz]', fontsize=24)
        plt.title('Amplitude Spectrum - Original Data')
        plt.xlim(0, fs / 2)  # Ensure frequency starts from 0

        # Compute FFT of the filtered data
        L_filtered = len(data_filtered)
        NFFT_filtered = 2 ** int(np.ceil(np.log2(L_filtered)))
        Y_filtered = fft(data_filtered, NFFT_filtered) / L_filtered
        f_filtered = fftfreq(NFFT_filtered, 1 / fs)[:NFFT_filtered // 2 + 1]

        # Time series plot (original and filtered data)
        plt.subplot(2, 2, 2)
        plt.plot(data, label='Original Data')
        plt.plot(data_filtered, 'r', label='Filtered Data')
        plt.xlabel('Time [ms]', fontsize=24)
        plt.ylabel('Amplitude [mV]', fontsize=24)
        plt.title('Time Series - Original and Filtered Data')
        plt.xlim(0, len(data))
        plt.legend()

        # Amplitude spectrum plot (filtered data)
        plt.subplot(2, 2, 4)
        if log_scale:
            plt.plot(f_filtered[0:-1], np.log10(2 * np.abs(Y_filtered[:NFFT_filtered // 2 + 1][0:-1])))
            plt.ylabel('log |Y(f)|', fontsize=24)
        else:
            plt.plot(f_filtered, 2 * np.abs(Y_filtered[:NFFT_filtered // 2 + 1]))
            plt.ylabel('|Y(f)|', fontsize=24)
        plt.xlabel('Frequency [Hz]', fontsize=24)
        plt.title('Amplitude Spectrum - Filtered Data')
        plt.xlim(0, fs / 2)  # Ensure frequency starts from 0

    plt.tight_layout()
    plt.show()

