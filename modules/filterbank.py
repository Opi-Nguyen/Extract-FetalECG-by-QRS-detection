import numpy as np
from scipy.signal import butter, sosfilt, iirnotch,lfilter

class FilterBank(object):
    def __init__(self, frequency_sampling: int = 1000, path=None):
        self.data_fs = frequency_sampling
        self.nyqs = frequency_sampling / 2 
    
    def butterworth_filter(
        self,
        data: np.ndarray,
        filter_type: str, # ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’  
        cutoff_freq : float,
        frequency_sampling: float = 1000,
        order: int = 4,
        ):
        """
        Apply a Butterworth high-pass/low-pass filter to the provided data.

        Parameters:
        data (array-like): The input signal data.
        filter_type (str): lowpass, highpass, bandpass, bandstop .
        cutoff_frequency (float): The cutoff frequency of the filter in Hz.
        sampling_rate (float): The sampling rate of the data in Hz.
        order (int): The order of the filter.

        Returns:
        filtered_data (array-like): The filtered signal data.
        """
        sos = butter(order, cutoff_freq, btype=filter_type, fs=frequency_sampling, output='sos')
        return sosfilt(sos, data)
        
    def notch_filter(self, frequency_sampling, bandwidth, freq_interest, Q, data):
        """
        Apply a notch (band-stop) filter to the provided data using iirnotch.

        Parameters:
        frequency_sampling (float): The sampling rate of the data in Hz.
        bandwidth (float): The bandwidth around the frequency of interest to be filtered out in Hz.
        freq_interest (float): The center frequency of the notch filter in Hz.
        Q (float): The quality factor of the notch filter.
        data (array-like): The input signal data to be filtered.

        Returns:
        filtered_data (array-like): The filtered signal data.
        """
        nyq = 0.5 * frequency_sampling
        f0 = freq_interest
        w0 = f0 / nyq
        bw = bandwidth / nyq
        b, a = iirnotch(w0, Q, fs=frequency_sampling)
        filtered_data = lfilter(b, a, data)
        return filtered_data

    