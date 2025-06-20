from scipy.signal import savgol_filter

def smooth_signal(data, window=11, poly=3):
    """
    Smooth a 1D signal using Savitzky-Golay filter.

    Parameters:
        data (list or np.array): Input signal to smooth
        window (int): Length of the filter window (must be odd and >= poly+2)
        poly (int): Order of the polynomial used to fit the samples

    Returns:
        list: Smoothed signal
    """
    if len(data) < window or window % 2 == 0:
        return data
    return savgol_filter(data, window_length=window, polyorder=poly).tolist()
