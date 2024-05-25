from modules.filterbank import FilterBank
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.linalg import svd



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
        return np.square(self.preprocessed_data)*(1e+10)
    
    def detect_QRS(self, data: np.ndarray, fs: float):
        window_size = int(0.12 * fs)  # 120 ms window
        integrated_ecg = np.convolve(data, np.ones(window_size)/window_size, mode='same')

        threshold = np.mean(integrated_ecg) + 0.5 * np.std(integrated_ecg)
        peaks, _ = find_peaks(integrated_ecg, height=threshold, distance=int(0.5 * fs))
        
        mean_peaks = np.sum(data[peaks])/len(peaks)
        threshold_peaks = mean_peaks/2.5
        filtered_peaks = peaks[data[peaks] > threshold_peaks]

        return filtered_peaks
    
    def PCA_ananlysis(
        self,
        data: np.array,
        QRS_data: list,
        start_index: any,
        P_Q_duration: any,
        total_cycles: any,
        cycle_width: any
        ):
        
        new_idx = start_index
        current_idx = start_index
        current_QRS = QRS_data[current_idx]
        
        if current_QRS < cycle_width * P_Q_duration:
            current_idx += 1
            new_idx = current_idx
        
        QRS_array = [[0 for _ in range(cycle_width)] for _ in range(total_cycles)]
        
        for i in range(total_cycles):
            current_QRS = QRS_data[current_idx]
            data_index = int(current_QRS - (cycle_width * P_Q_duration))
            for j in range(cycle_width):
                QRS_array[i][j] = data[data_index]
                data_index += 1
            current_idx += 1

        return QRS_array, new_idx
<<<<<<< HEAD
    def SVD_ananlysis(
        self,
        data: np.array,
        QRS_data: list,
        start_index: any,
        P_Q_duration: any,
        total_cycles: any,
        cycle_width: any
        ):
        
        new_idx = start_index
        current_idx = start_index
        current_QRS = QRS_data[current_idx]
        
        if current_QRS < cycle_width * P_Q_duration:
            current_idx += 1
            new_idx = current_idx
        
        QRS_array = [[0 for _ in range(cycle_width)] for _ in range(total_cycles)]
        
        for i in range(total_cycles):
            current_QRS = QRS_data[current_idx]
            data_index = int(current_QRS - (cycle_width * P_Q_duration))
            for j in range(cycle_width):
                QRS_array[i][j] = data[data_index]
                data_index += 1
            current_idx += 1

        return QRS_array, new_idx
=======
    
    
    def SVD_ananlysis(
        self,
        PCA_array: np.array,
        ):
        U, sigma, VT = svd(PCA_array)
        return U, sigma, VT
>>>>>>> 2908326 (add data, fix storage csv path)
    
    def recontruct_MECG(self):
        pass
    
    def subtract_template(self):
        pass
    
    def filter_local_extrema(self, data: np.ndarray, local_extremas: any, acceptance_width: float):
        i = 0
        print(local_extremas)
        while (i < len(local_extremas[0]) - 1):
            if (local_extremas[0][i+1] - local_extremas[0][i]) <= acceptance_width:
                if data[local_extremas[0][i+1]] <= data[local_extremas[0][i]]:
                    local_extremas = np.delete(local_extremas, [i+1], 1)
                else:
                    local_extremas = np.delete(local_extremas, [i], 1)
            i +=1
        return local_extremas
    
    