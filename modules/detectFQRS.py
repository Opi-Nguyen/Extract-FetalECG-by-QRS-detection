from modules.filterbank import FilterBank
import numpy as np


class DetectFQRS(object):
    def __init__(self, frequency_sampling: float, data: np.ndarray, **kwargs):
        self.sample_rate = frequency_sampling
        self.filter_bank = FilterBank(frequency_sampling)
        range = kwargs.get('range', None)
        if range is not None:
            self.data = data[range[0]: range[1]]
        else:
            self.data = data        
    
    def preprocess(
        self, 
        data: np.ndarray,
        cutoff_low: float,
        cutoff_high: float,
        filter_order: int,
        ):
        #notched-pass filter
        notch_filtered_data = self.filter_bank.notch_filter(
            frequency_sampling=self.sample_rate,
            bandwidth=20,
            freq_interest=50,
            Q=30,
            data=data,
        )
        notch_filtered_data = self.filter_bank.notch_filter(
            frequency_sampling=self.sample_rate,
            bandwidth=5,
            freq_interest=150,
            Q=30,
            data=notch_filtered_data,
        )
        #low-pass filter
        lowpass_filtered_data = self.filter_bank.butterworth_filter(
            data=notch_filtered_data,
            filter_type="lowpass",
            cutoff_freq=cutoff_low,
            order=filter_order
        )
        #high-pass filter
        highpass_filtered_data = self.filter_bank.butterworth_filter(
            data=lowpass_filtered_data,
            filter_type="highpass",
            cutoff_freq=cutoff_high,
            order=filter_order
        )
        self.preprocessed_data = highpass_filtered_data
        return np.square(self.preprocessed_data)
    
    def detect_QRS(self):
        pass
    
    def analyze_PCA(self):
        pass
    
    def recontruct_MECG(self):
        pass
    
    def subtract_template(self):
        pass