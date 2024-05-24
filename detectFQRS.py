from filterbank import FilterBank
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
    
    def preprocess(self, data: np.ndarray):
        #notched-pass filter
        notch_filtered_data = self.filter_bank.notch_filter(
            frequency_sampling=self.sample_rate,
            bandwidth=20,
            freq_interest=50,
            data=data,
        )
        notch_filtered_data = self.filter_bank.notch_filter(
            frequency_sampling=self.sample_rate,
            bandwidth=5,
            freq_interest=150,
            data=data,
        )
        #high-pass filter
        #low-pass filter
        pass
    
    
    def detect_QRS(self):
        pass
    
    def analyze_PCA(self):
        pass
    
    def recontruct_MECG(self):
        pass
    
    def subtract_template(self):
        pass