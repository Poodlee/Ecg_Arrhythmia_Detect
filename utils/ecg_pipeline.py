import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ECG signal pre-processing (denoising, standardization, feature extraction, etc.)

def bandpass_filter(signal_array, fs, lowcut=0.5, highcut=40, order=4):
    """
    Apply Butterworth bandpass filter to ECG signals.
    
    Args:
        signal_array: numpy array of shape (N,) or (N, T)
        fs: sampling frequency
        lowcut: low cutoff frequency in Hz
        highcut: high cutoff frequency in Hz
        order: filter order
        
    Returns:
        Filtered signal (same shape as input)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    if signal_array.ndim == 1:
        return signal.filtfilt(b, a, signal_array)
    else:
        return np.array([signal.filtfilt(b, a, sig) for sig in signal_array])


def detrend_signal(signal_array):
    """
    Remove linear trend from signal.
    
    Returns:
        Detrended signal
    """
    if signal_array.ndim == 1:
        return signal.detrend(signal_array)
    else:
        return signal.detrend(signal_array, axis=-1)


def standardize_signal(signal_array):
    """
    Apply Min-Max normalization to ECG signals.
    
    Returns:
        Standardized signal
    """

    def min_max(sig):
        min_val = np.min(sig)
        max_val = np.max(sig)

        if max_val - min_val == 0:
            return np.zeros_like(sig)
        return (sig - min_val) / (max_val - min_val)
    
    if signal_array.ndim == 1:
        return min_max(signal_array)
    else:
        return np.array([min_max(sig) for sig in signal_array])
    
def stockwell_transform(signal, fs, fmin=0, fmax=None):
    from stockwell import st
    """
    Apply Stockwell Transform (S-transform) to a 1D signal.

    Args:
        signal (np.ndarray): 1D array of the signal
        fs (float): Sampling frequency in Hz
        fmin (float): Minimum frequency (Hz) for transform
        fmax (float or None): Maximum frequency (Hz) for transform.
                              If None, defaults to fs/2.

    Returns:
        st_result (np.ndarray): 2D array of complex S-transform (freq x time)
        freqs (np.ndarray): Frequency axis values
        times (np.ndarray): Time axis values
    """
    N = len(signal)
    duration = N / fs
    t = np.linspace(0, duration, N)

    df = 1.0 / duration
    if fmax is None:
        fmax = fs / 2

    fmin_samples = int(fmin / df)
    fmax_samples = int(fmax / df)

    # Apply Stockwell Transform
    st_result = st.st(signal, fmin_samples, fmax_samples)

    freqs = np.linspace(fmin, fmax, fmax_samples - fmin_samples)
    return st_result, freqs, t
    