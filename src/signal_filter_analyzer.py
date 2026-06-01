import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, lfilter, freqz

class SignalFilterAnalyzer:
    """
    This Class used to analysis data in frequency domain, and compare the original signal with the filtered signal in
    both time and frequency domains. Also possible to compare different designed filters.

    last modified: 18.05.2026 XW
    usage start from: SoSe2026
    """

    def __init__(self, data, fs):
        """
        Initialize the analyzer.
        :param data: Original 1D signal data
        :param fs: Sampling frequency (Hz)
        """
        self.data = np.array(data)
        self.fs = fs
        self.L = len(self.data)

        # Variables reserved for the filter and processing results
        self.data_filtered = None
        self.b = None
        self.a = None
        self.w = None
        self.h = None
        self.order = None
        self.cutoff = None
        self.btype = None
        self.filter_method = None

    def apply_filter(self, order, cutoff, btype='low', filter_method='filtfilt'):

        """
        Design and apply the filter.
        """

        self.order = order
        self.cutoff = cutoff
        self.btype = btype
        self.filter_method = filter_method

        # Design the filter
        self.b, self.a = butter(order, cutoff, btype=btype, fs=self.fs)

        # Apply the filter
        if filter_method == 'filtfilt':
            self.data_filtered = filtfilt(self.b, self.a, self.data)
        elif filter_method == 'lfilter':
            self.data_filtered = lfilter(self.b, self.a, self.data)
        else:
            raise ValueError("filter_method must be either 'filtfilt' or 'lfilter'")

        # Calculate the frequency response of the filter and save it for plotting
        self.w, self.h = freqz(self.b, self.a, worN=8000, fs=self.fs)

        return self.data_filtered

    def _add_cutoff_lines(self, ax, color='red'):

        """
        Internal helper function: draw vertical dashed lines for cutoff frequencies.
        """

        cutoffs = self.cutoff if isinstance(self.cutoff, (list, tuple, np.ndarray)) else [self.cutoff]
        for c in cutoffs:
            ax.axvline(x=c, color=color, linestyle='--', linewidth=1.5, label=f'Cutoff: {c} Hz')

        # Deduplicate legend handles
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

    def plot_spectrum(self, log_scale=False):

        """
        Plot the time and frequency domain comparisons between original and filtered signals (2x2 layout).
        """

        if self.data_filtered is None:
            raise RuntimeError("Please run apply_filter() before plotting the spectrum.")

        NFFT = 2 ** int(np.ceil(np.log2(self.L)))
        # f = fftfreq(NFFT, 1 / self.fs)[:NFFT // 2 + 1]
        f = fftfreq(NFFT, 1 / self.fs)[:NFFT // 2]

        Y_orig = fft(self.data, NFFT) / self.L
        Y_filt = fft(self.data_filtered, NFFT) / self.L

        # amp_orig = 2 * np.abs(Y_orig[:NFFT // 2 + 1])
        # amp_filt = 2 * np.abs(Y_filt[:NFFT // 2 + 1])
        amp_orig = 2 * np.abs(Y_orig)[:NFFT // 2]
        amp_filt = 2 * np.abs(Y_filt)[:NFFT // 2]

        if log_scale:
            amp_orig = np.log10(amp_orig + 1e-12)
            amp_filt = np.log10(amp_filt + 1e-12)

        t = np.arange(self.L) * (1000 / self.fs)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Signal Analysis: Original vs Filtered', fontsize=16)

        # Top-left: Original time series
        axs[0, 0].plot(t, self.data, color='blue')
        axs[0, 0].set_title('Time Series - Original')
        axs[0, 0].set_xlabel('Time [ms]')
        axs[0, 0].set_ylabel('Amplitude')

        # Top-right: Time series comparison
        axs[0, 1].plot(t, self.data, color='blue', alpha=0.3, label='Original')
        axs[0, 1].plot(t, self.data_filtered, color='red', label='Filtered')
        axs[0, 1].set_title('Time Series - Overlay')
        axs[0, 1].set_xlabel('Time [ms]')
        axs[0, 1].legend()

        # Bottom-left: Original spectrum
        axs[1, 0].plot(f, amp_orig, color='blue')
        axs[1, 0].set_title('Spectrum - Original')
        axs[1, 0].set_xlabel('Frequency [Hz]')
        axs[1, 0].set_ylabel('log |Y(f)|' if log_scale else '|Y(f)|')
        axs[1, 0].set_xlim(0, self.fs / 2)
        self._add_cutoff_lines(axs[1, 0])

        # Bottom-right: Spectrum comparison
        axs[1, 1].plot(f, amp_orig, color='blue', alpha=0.3, label='Original')
        axs[1, 1].plot(f, amp_filt, color='red', label='Filtered')
        axs[1, 1].set_title('Spectrum - Overlay')
        axs[1, 1].set_xlabel('Frequency [Hz]')
        axs[1, 1].set_xlim(0, self.fs / 2)
        self._add_cutoff_lines(axs[1, 1])

        plt.tight_layout()
        plt.show()

    def plot_filter_characteristics(self):

        """
        Plot the filter's Gain and Phase characteristics (Independent window, top/bottom layout).
        """

        if self.w is None or self.h is None:
            raise RuntimeError("Please run apply_filter() before plotting filter characteristics.")

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Filter Characteristics ({self.btype}, Order: {self.order}, Method: {self.filter_method})', fontsize=14)

        # Top: Gain
        axs[0].plot(self.w, np.abs(self.h), color='purple', linewidth=2, label='Gain')
        axs[0].set_ylabel("Gain")
        axs[0].grid(True)
        self._add_cutoff_lines(axs[0], color='forestgreen')

        # Bottom: Phase
        axs[1].plot(self.w, np.angle(self.h), color='purple', linewidth=2, label='Phase')
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].set_ylabel("Phase Shift [Radians]")
        axs[1].grid(True)
        self._add_cutoff_lines(axs[1], color='forestgreen')

        plt.tight_layout()
        plt.show()

    def plot_zoomed_comparison(self, data_dict, time_interval, marked_positions=None):

        """
        A generic zoomed-in comparison plot, used to compare the original signal with any set of filtered signals.

        :param data_dict: Dictionary where keys are legend labels and values are 1D data arrays.
                          Example: {'Order 2': data_2, 'Order 8': data_8}
        :param time_interval: List or tuple setting the X-axis display range, e.g., [280, 310]
        :param marked_positions: Number or list, to draw vertical dashed lines marking key time points, e.g., 285 or [285, 290]
        """

        t = np.arange(self.L) * (1000 / self.fs)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Signal Comparison (Zoomed In)", fontsize=16, fontweight='bold', pad=15)

        # 1. Always plot the original signal as the reference baseline
        ax.plot(t, self.data, color='royalblue', alpha=0.3, label='Original Data', linewidth=2)

        # 2. Iterate and plot all comparison data provided
        # Predefine a few high-contrast colors and linestyles for the loop
        colors = ['red', 'darkorange', 'forestgreen', 'purple']
        linestyles = ['-', '--', '-.', ':']

        for i, (label, data) in enumerate(data_dict.items()):
            c = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            ax.plot(t, data, color=c, linestyle=ls, label=label, linewidth=2)

        # Lock the local time interval (Zoom In)
        ax.set_xlim(time_interval[0], time_interval[1])
        ax.set_xlabel('Time [ms]', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)

        # Process the marker lines
        if marked_positions is not None:
            if isinstance(marked_positions, (int, float)):
                marked_positions = [marked_positions]

            for pos in marked_positions:
                if time_interval[0] <= pos <= time_interval[1]:
                    ax.axvline(x=pos, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                    ax.text(pos, ax.get_ylim()[1]*0.95, f'{pos} ms', color='black',
                            ha='right', va='top', rotation=90, fontsize=10,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()