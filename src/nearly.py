import numpy as np


def nearly(tim, timeArray):
    """
    [sample] = nearly(tim, timeArray)

    PURPOSE:
    Find closest sample index to the time
    in tim and return the index.
    INPUT:
    tim:       Target time point(s). Can be a vector of target time points.
    timeArray: A vector holding a time basis e.g. of an ECoG measurement
               This is assumed to be a montonically increasing series of
               numbers.
    OUTPUT:
    sample:    The index to the sample closest to the targeted time point
    EXAMPLE:
    timeArray = np.arange(1, 101)  # equivalent to MATLAB 1:100
    tim = np.array([-100, 2.4, 105])
    sample = nearly(tim, timeArray)
    sample should hold [0, 2, 99]  # (Python 0-indexed; add 1 for MATLAB indexing)
    """
    tim = np.array([tim]) if np.isscalar(tim) else tim
    sample = np.zeros(tim.shape, dtype=int)
    tmp = np.zeros((tim.shape[0], 2), dtype=int)
    for k in range(len(tim)):
        # Find the first index where timeArray > tim[k]
        idx = np.where(timeArray > tim[k])[0]
        if idx.size > 0:
            tmp[k, 0] = idx[0]
        else:
            tmp[k, 0] = np.argmax(timeArray)  # If nothing is larger, choose the max index

        # Find the last index where timeArray <= tim[k]
        idx = np.where(timeArray <= tim[k])[0]
        if idx.size > 0:
            tmp[k, 1] = idx[-1]
        else:
            tmp[k, 1] = np.argmin(timeArray)  # If nothing is smaller, choose the min index

        # Find the closest index between the two candidates
        diff1 = abs(timeArray[tmp[k, 0]] - tim[k])
        diff2 = abs(timeArray[tmp[k, 1]] - tim[k])
        sample[k] = tmp[k, 0] if diff1 <= diff2 else tmp[k, 1]

    # If input was scalar, return scalar output instead of array
    return sample[0] if sample.size == 1 else sample


    # for k in range(len(tim)):
    #     # t = min(find(timeArray > tim(k)));
    #     idx = np.where(timeArray > tim[k])[0]
    #     if idx.size > 0:
    #         tmp[k, 0] = idx[0]
    #     else:
    #         # [y, idx] = max(timeArray);
    #         idx_max = np.argmax(timeArray)
    #         tmp[k, 0] = idx_max  # nothing is larger, we choose the max
    #
    #     # t = max(find(timeArray <= tim(k)));
    #     idx = np.where(timeArray <= tim[k])[0]
    #     if idx.size > 0:
    #         tmp[k, 1] = idx[-1]
    #     else:
    #         # [y, idx] = min(timeArray);
    #         idx_min = np.argmin(timeArray)
    #         tmp[k, 1] = idx_min  # nothing is smaller, we choose the min
    #
    #     # [y, idx] = min([abs(timeArray(tmp(k,1))-tim(k)),abs(timeArray(tmp(k,2))-tim(k))]);
    #     diff1 = abs(timeArray[tmp[k, 0]] - tim[k])
    #     diff2 = abs(timeArray[tmp[k, 1]] - tim[k])
    #     if diff1 <= diff2:
    #         chosen = 0
    #     else:
    #         chosen = 1
    #     sample[k] = tmp[0][k, chosen]
    # return sample
