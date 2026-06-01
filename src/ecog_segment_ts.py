import numpy as np


def ecog_segment_ts(ecog, trigger_idx, pre_dur_samp, post_dur_samp):
    """
    Segment time series data around the indices in trigger_idx.

    Parameters:
    -----------
    ecog : dict or numpy.ndarray
        An ecog structure (dictionary) or a matrix of data time series with time along columns.
    trigger_idx : list or numpy.ndarray
        A vector of indices marking the segment start in the first dimension of the time series.
    pre_dur_samp : int
        The number of pre-trigger samples in the segment extracted.
    post_dur_samp : int
        The number of post-trigger samples in the segment extracted.

    Returns:
    --------
    ecog : dict or numpy.ndarray
        An ecog structure with data segments in the 'data' field (if input is a dictionary),
        or a 3D numpy array containing the segments (if input is a numpy array).
    bas : numpy.ndarray, optional
        The baseline (mean of pre-trigger samples) subtracted from each segment.
        Only returned if the input is a numpy array (legacy support).
    """

    if isinstance(ecog, dict):
        # If ecog is a dictionary (ecog structure)
        ecog['data'] = np.array(ecog['data'])
        ecog['refChanTS'] = np.array(ecog['refChanTS'])

        n_channels = ecog['data'].shape[0]
        seg_length = pre_dur_samp + post_dur_samp
        n_segments = len(trigger_idx)

        # Initialize segmented data
        seg = np.zeros((n_channels, seg_length, n_segments))
        for k in range(n_segments):
            start_idx = trigger_idx[k] - pre_dur_samp
            end_idx = trigger_idx[k] + post_dur_samp
            seg[:, :, k] = ecog['data'][:, start_idx:end_idx]

        # Update ecog structure
        ecog['data'] = seg

        # Handle reference channel time series (if present)
        if 'refChanTS' in ecog and ecog['refChanTS'] is not None:
            if len(ecog['refChanTS'].shape) == 1:
                ref_seg = np.zeros((1, seg_length, n_segments))
            else:
                ref_seg = np.zeros((ecog['refChanTS'].shape[0], seg_length, n_segments))
            for k in range(n_segments):
                start_idx = trigger_idx[k] - pre_dur_samp
                end_idx = trigger_idx[k] + post_dur_samp
                ref_seg[:, :, k] = ecog['refChanTS'][start_idx:end_idx]
            ecog['refChanTS'] = ref_seg

        # Handle trigger time series (if present)
        if 'triggerTS' in ecog:
            trigger_seg = np.zeros((ecog['triggerTS'].shape[0], seg_length, n_segments))
            for k in range(n_segments):
                start_idx = trigger_idx[k] - pre_dur_samp
                end_idx = trigger_idx[k] + post_dur_samp
                trigger_seg[:, :, k] = ecog['triggerTS'][:, start_idx:end_idx]
            ecog['triggerTS'] = trigger_seg

        # Update ecog metadata
        ecog['nSamp'] = seg_length
        ecog['nBaselineSamp'] = pre_dur_samp
        ecog['timebase'] = np.arange(-pre_dur_samp, post_dur_samp) * ecog['sampDur']

        return ecog

    else:
        # If ecog is a numpy array (legacy support)
        n_samples, n_channels = ecog.shape
        seg_length = pre_dur_samp + post_dur_samp
        n_segments = len(trigger_idx)

        # Initialize segmented data
        seg = np.zeros((seg_length, n_channels, n_segments))
        for k in range(n_segments):
            start_idx = trigger_idx[k] - pre_dur_samp
            end_idx = trigger_idx[k] + post_dur_samp
            seg[:, :, k] = ecog[start_idx:end_idx, :]

        # Subtract the baseline (mean of pre-trigger samples)
        bas = np.mean(seg[:pre_dur_samp, :, :], axis=0)
        seg = seg - np.tile(bas, (seg_length, 1, 1))

        return seg, bas