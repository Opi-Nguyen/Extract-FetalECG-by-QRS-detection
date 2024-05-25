from modules.filterbank import FilterBank
from modules.detectFQRS import DetectFQRS
import numpy as np
import matplotlib.pyplot as plt

#get data
raw_direct_fecg_data, raw_abdecg_data_c1, _, _, _ = np.loadtxt('csv_data/r04.csv', delimiter=',', unpack=True).tolist()
data = np.asarray(raw_abdecg_data_c1)
crop_data = data[5000:55000]

#config params, attr
frequency_sampling = 1000
FQRS_detector = DetectFQRS(frequency_sampling=frequency_sampling, data=crop_data)

#1 preprocess
preprocessed_data = FQRS_detector.preprocess(data=crop_data, cutoff_low=15, cutoff_high=2, filter_order=2)

#2 detectQRS
Meternal_QRS = FQRS_detector.detect_QRS(data=preprocessed_data, fs=200)

##MQRS template subtraction
cycle_width = 700
P_Q_duration = 0.25
start_idx = 0
while len(Meternal_QRS)-start_idx > 20:
    current_Meternal_QRS = Meternal_QRS[start_idx]
    if (current_Meternal_QRS < cycle_width * P_Q_duration)
        start_idx +=1
    
    
    
#3 PCA
PCA_QRS_array, new_idx = FQRS_detector.PCA_ananlysis

#4 SVD n Cycle

#MECG 










time_len = len(crop_data)
t = np.linspace(0, time_len/1000, int(time_len), endpoint=False)
print(len(t))


plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t, crop_data, label='Original signal')
plt.title('Original Signal')
plt.xlabel('Time [seconds]')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, preprocessed_data, label='Filtered signal (high-pass)', color='orange')
plt.plot(max_local_extremas/frequency_sampling, preprocessed_data[max_local_extremas], 'ro', label='Relative Maxima')
plt.title('Filtered Signal')
plt.xlabel('Time [seconds]')
plt.grid()
plt.tight_layout()
plt.show()