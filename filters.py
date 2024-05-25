from scipy.signal import butter, filtfilt
from scipy import signal
from scipy.signal import iirfilter
import numpy as np



def butter_lowpass_filter(data, cutoff_frequency, sampling_rate, order):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_frequency / nyqs
    b,a = butter(order, normal_cutoff_freq, 'low', False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def butter_highpass_filter(data, cutoff_frequency, sampling_rate, order):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_frequency / nyqs
    b,a = butter(order, normal_cutoff_freq, 'high', False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def Implement_Notch_Filter(sampling_frequency, bandwidth, freq_interest, ripple, order, filter_type, data):
    nyq  = 0.5*sampling_frequency
    low  = (freq_interest - bandwidth/2.0)/nyq
    high = (freq_interest + bandwidth/2.0)/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def derivative_filter(data):
    derivative = np.zeros(len(data))
    for i in range(2,len(data) - 3):
        derivative[i] = (-2*data[i-2] - data[i-1] + data[i+1] + 2*data[i+2])/10
    return derivative